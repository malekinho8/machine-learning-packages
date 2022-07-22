# # Bathymetry Data Set Utilities

# +
# Proprietary and Confidential
# Copyright 2003 - 2020 Stone Aerospace
# 3511 Caldwell Lane, Del Valle, TX 78617
# All rights reserved
# Author: Malek Ibrahim
# Date: 06.21.2022
# -

import numpy as np
from PIL import Image
import numpy as np
import rasterio
from osgeo import gdal
from rasterio.plot import show
import matplotlib.pyplot as plt
from rasterio.warp import transform_bounds
from torchvision import transforms
import cv2
import os
import pandas as pd


# ### Overview
# Note, this function definition is based heavily on bathymetry_create_data_1ab.py and MBARI_load_25-m_data_test_1a.py. For step by step demonstration of this function, see those files.

def load_bathymetry_map(file_path, transform_flag=None):
    if '.tif' in file_path:
        img = rasterio.open(file_path)
        full_img = img.read(1, masked=True)
        
        try:
            if transform_flag is None:
                wgs84_bounds = transform_bounds(img.crs, "epsg:4326", *img.bounds) # make sure projection is correct
                transform = rasterio.transform.from_bounds(*wgs84_bounds,img.shape[1],img.shape[0])
                cols, rows = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
                xs, ys = rasterio.transform.xy(transform, rows, cols)
                lons = np.array(xs)[0,:]
                lats = np.array(ys)[:,0]
            elif transform_flag is 'linear':
                # get boundaries and initialize variables
                wgs84_bounds = transform_bounds(img.crs, "epsg:4326", *img.bounds)
                xllcorner = wgs84_bounds[0]
                yllcorner = wgs84_bounds[1]
                div_factor = 89,302.4562 # determined this from montereyc.asc file, 25 m / x = 0.0002799475072
                temp = gdal.Open(file_path)
                gt = temp.GetGeoTransform()
                cellsize_x = np.abs(gt[1])/div_factor
                cellsize_y = np.abs(gt[-1])/div_factor
                nrows = img.shape[0]
                ncols = img.shape[1]
                
                # create lat long vectors
                lons = np.linspace(start=xllcorner, stop=wgs84_bounds[2], num=int(ncols))
                lats = np.linspace(start=wgs84_bounds[3], stop=yllcorner, num=int(nrows)) # we have to reverse this linspace because of the way the data is formatted; (0,0) pixel is in the upper right corner
                
                
        except:
            lons = None
            lats = None
            
    elif '.asc' in file_path:
        img  = np.loadtxt(file_path, skiprows=6)
        info = pd.read_table(file_path,nrows=6,header=None, delimiter=' ')
        
        # define the variables presented in the table above explicitly
        ncols = info.iloc[0,1]
        nrows = info.iloc[1,1]
        xllcorner = info.iloc[2,1]
        yllcorner = info.iloc[3,1]
        cellsize = info.iloc[4,1]
        nanvalue = info.iloc[5,1]
        
        # define latitude/longitude vectors
        lons = np.linspace(start=xllcorner, stop=ncols*cellsize + xllcorner, num=int(ncols))
        lats = np.linspace(start=nrows*cellsize + yllcorner, stop=yllcorner, num=int(nrows)) # we have to reverse this linspace because of the way the data is formatted; (0,0) pixel is in the upper right corner
        
        # define masked array according to nanvalue
        full_img = np.ma.masked_where(img == nanvalue, img)
    else:
        img = 'unknown file type'
        lons = None
        lats = None
    
    output = {
        'img': full_img,
        'lats': lats,
        'lons': lons
    }
    
    return output

def zm_to_8bit(data):
    """
    This function takes in data bound within [-1, 1] and transforms it to [0, 256) so it can be saved as an 8 bit image file (reduce data set size).

    Args:
        data: N-D array of floats centered around zero and bounded within range -1 to 1

    Returns:
        out_8bit: N-D array of uint8's ranging from 0 to 255

    """

    temp = (data + 1)/2
    out_8bit = (temp*255).astype(np.uint8)

    return out_8bit

def bit_to_zm(data):
    """
    This function takes in data bound within [0, 255) and centers it around zero such that is bound by [-1, 1].

    Args:
        data: N-D array of uint8's ranging from 0 to 255

    Returns:
        out_8bit: N-D array of floats centered around zero and bounded within range -1 to 1

    """

    temp = (data + 1)/2
    out_8bit = (temp*255).astype(np.uint8)

    return out_8bit

# ### Gridder Class
#
# The goal for this class is to allow for various methods such as creating a gridded data set and saving array data to a folder, determining the distribution of zero-mean data samples, and potentially converting data to an image format that could be saved in an image folder, along with the transformations that allow the user to return the data to the true range.

class Gridder():
    
    def __init__(self, full_img, lats, lons, window_size, overlap, im_dir, prefix=None):
        """
        The Gridder class, which contains various methods including the ability to create a gridded data set,
        plot a histogram of sample values, and more...

        Args:
            full_img: masked np array of shape (height,width) corresponding to fully loaded bathymetry data
            lats: 1D vector of floats, latitude coordinates of data in full_img
            lons: 1D vector of floats, longitude coordinates of data in full_img
            window_size: int, dimension of the output grids
            overlap: float in the range [0, 1), how much the grids should overlap with one another
            im_dir: str, path to save images
            prefix: (optional) str, what to include when saving each file            

        """
        
        self.full_img = full_img
        self.lats = lats
        self.lons = lons
        self.window_size = window_size
        self.overlap = overlap
        self.im_dir = im_dir
        self.prefix = prefix


    def grid_zm_scaled_npy(self, scale_factor):
        """
        Saves scaled zero-mean gridded data set as numpy arrays from a bathymetry map.

        Args:
            self: Gridder object, (see __init__ for variables)     
            scale_factor: float, how much to divide all samples in the data set by in order to put into range -1 to 1

        """

        # define the prefix for file saving
        prefix = 'im' if self.prefix is None else self.prefix

        # define the height and width of the fully loaded map
        height = self.full_img.shape[0]
        width = self.full_img.shape[1]

        # determine the step size based on the window size and overlap %
        step_size = self.window_size - int(np.floor(self.window_size*self.overlap))

        # initialize row index and column index to iterate over
        r_idx = 0
        c_idx = 0
        i = 0

        # begin the main loop
        while r_idx + self.window_size <= height:
            while c_idx + self.window_size <= width:
                # define the grid
                box = self.full_img[r_idx:(r_idx+self.window_size),c_idx:(c_idx+self.window_size)]
                lat = np.mean(self.lats[r_idx:(r_idx+self.window_size)]) # average latitude coordinate of the data sample
                lon = np.mean(self.lons[c_idx:(c_idx+self.window_size)]) # average longitude coordinate of the sample
                
                # filter out data that contains gaps
                if ~np.any(box.mask): # if there are not any masked entries
                    if not os.path.exists(os.path.join(self.im_dir, f'{prefix}_n-{i:06d}_lat-{lat:.4f}_lon-{lon:.4f}.npy')):
                        # zero-mean the data
                        box_zm = (box - np.median(box)).astype(np.float32)
                        
                        # save the data
                        np.save(os.path.join(self.im_dir, f'{prefix}_n-{i:06d}_lat-{lat:.4f}_lon-{lon:.4f}.npy'), box_zm.data/scale_factor)
                    
                    # increment the data counter
                    i += 1

                # increment the column index
                c_idx += step_size

            # increment the row index, restart the column index back to the left hand side
            r_idx += step_size
            c_idx = 0

        print(f'Total Samples: {i}')

    def grid_zm_scaled_png(self, scale_factor):
        """
        Saves scaled zero-mean gridded data set as png's from a bathymetry map.

        Args:
            self: Gridder object, (see __init__ for variables)     
            scale_factor: float, how much to divide all samples in the data set by in order to put into range -1 to 1

        """

        # define the prefix for file saving
        prefix = 'im' if self.prefix is None else self.prefix

        # define the height and width of the fully loaded map
        height = self.full_img.shape[0]
        width = self.full_img.shape[1]

        # determine the step size based on the window size and overlap %
        step_size = self.window_size - int(np.floor(self.window_size*self.overlap))

        # initialize row index and column index to iterate over
        r_idx = 0
        c_idx = 0
        i = 0

        # begin the main loop
        while r_idx + self.window_size <= height:
            while c_idx + self.window_size <= width:
                # define the grid
                box = self.full_img[r_idx:(r_idx+self.window_size),c_idx:(c_idx+self.window_size)]
                lat = np.mean(self.lats[r_idx:(r_idx+self.window_size)]) # average latitude coordinate of the data sample
                lon = np.mean(self.lons[c_idx:(c_idx+self.window_size)]) # average longitude coordinate of the sample
                
                # filter out data that contains gaps
                if ~np.any(box.mask): # if there are not any masked entries
                    if not os.path.exists(os.path.join(self.im_dir, f'{prefix}_n-{i:06d}_lat-{lat:.4f}_lon-{lon:.4f}.png')):
                        # zero-mean the data, clip to -1 to 1
                        box_zm = zm_to_8bit(np.clip((box - np.median(box)).astype(np.float32)/scale_factor, -1, 1))
                        save_data = transforms.ToPILImage()(box_zm.data)
                        
                        # save the data
                        save_data.save(os.path.join(self.im_dir, f'{prefix}_n-{i:06d}_lat-{lat:.4f}_lon-{lon:.4f}.png'))
                    
                    # increment the data counter
                    i += 1

                # increment the column index
                c_idx += step_size

            # increment the row index, restart the column index back to the left hand side
            r_idx += step_size
            c_idx = 0

        print(f'Total Samples: {i}')

    def grid_zm_scaled_chunk(self, scale_factor):
        """
        Obtains a chunk of gridded zero-mean data, scaled by scale factor

        Args:
            self: Gridder object, (see __init__ for variables)     
            scale_factor: float, how much to divide all samples in the data set by in order to put into range -1 to 1

        Returns:
            chunk: gridded masked numpy array of shape (N, self.window_size, self.window_size, ...)
            means: the median value of each window in the chunk, this will be needed to re-transform the data output from the diffusion model for viewing
            R: number of rows of images (for indexing purposes)
            C: number of cols of images

        """

        # define the prefix for file saving
        prefix = 'im' if self.prefix is None else self.prefix

        # define the height and width of the fully loaded map
        height = self.full_img.shape[0]
        width = self.full_img.shape[1]

        # determine the step size based on the window size and overlap %
        step_size = self.window_size - int(np.floor(self.window_size*self.overlap))

        # initialize row index and column index to iterate over
        r_idx = 0
        c_idx = 0
        i = 0

        # initialize the output
        chunk = []
        means = []

        # begin the main loop
        while r_idx + self.window_size <= height:
            while c_idx + self.window_size <= width:
                # define the grid
                box = self.full_img[r_idx:(r_idx+self.window_size),c_idx:(c_idx+self.window_size)]
                lat = np.mean(self.lats[r_idx:(r_idx+self.window_size)]) # average latitude coordinate of the data sample
                lon = np.mean(self.lons[c_idx:(c_idx+self.window_size)]) # average longitude coordinate of the sample
                
                # zero-mean the data
                if not np.all(box.mask):
                    med_temp = np.median(box[~box.mask])
                else:
                    med_temp = np.nan

                box_zm = (box - med_temp)/scale_factor
                box_zm.data[box.mask] = np.random.randn(len(box_zm.data[box.mask]))
                
                # output data to list
                chunk.append(box_zm)
                means.append(med_temp)
            
                # increment the data counter
                i += 1

                # increment the column index
                c_idx += step_size

            # increment the row index, restart the column index back to the left hand side
            r_idx += step_size
            c_idx = 0
        
        # count the number of rows/cols 
        R = C = i = j = 0
        while i + self.window_size <= height:
            i += step_size
            R += 1
        
        while j + self.window_size <= width:
            # increment the column index
            j += step_size
            C += 1

        return np.ma.array(chunk), means, R, C
    
    def grid_simple_png(self):
        """
        Obtains gridded samples and normalizes each individually to the range of 0 to 255 so that they can be saved as images using cv2.imwrite().

        Args:
            self: Gridder object, (see __init__ for variables)     

        """

        # define the prefix for file saving
        prefix = 'im' if self.prefix is None else self.prefix

        # define the height and width of the fully loaded map
        height = self.full_img.shape[0]
        width = self.full_img.shape[1]

        # determine the step size based on the window size and overlap %
        step_size = self.window_size - int(np.floor(self.window_size*self.overlap))

        # initialize row index and column index to iterate over
        r_idx = 0
        c_idx = 0
        i = 0

        # begin the main loop
        while r_idx + self.window_size <= height:
            while c_idx + self.window_size <= width:
                # define the grid
                box = self.full_img[r_idx:(r_idx+self.window_size),c_idx:(c_idx+self.window_size)]
                lat = np.mean(self.lats[r_idx:(r_idx+self.window_size)]) # average latitude coordinate of the data sample
                lon = np.mean(self.lons[c_idx:(c_idx+self.window_size)]) # average longitude coordinate of the sample
                
                # filter out data that contains gaps
                if ~np.any(box.mask): # if there are not any masked entries
                    if not os.path.exists(os.path.join(self.im_dir, f'{prefix}_n-{i:06d}_lat-{lat:.4f}_lon-{lon:.4f}.png')):
                        # zero-mean the data, clip to -1 to 1
                        temp = box - box.min() # translate data minimum to zero
                        box_data = ((temp / temp.max())*255).astype(np.uint8)
                        save_data = transforms.ToPILImage()(box_data)
                        
                        # save the data
                        save_data.save(os.path.join(self.im_dir, f'{prefix}_n-{i:06d}_lat-{lat:.4f}_lon-{lon:.4f}.png'))
                    
                    # increment the data counter
                    i += 1

                # increment the column index
                c_idx += step_size

            # increment the row index, restart the column index back to the left hand side
            r_idx += step_size
            c_idx = 0

        print(f'Total Samples: {i}')

        
    def histogram_zm(self):
        """
        Creates a distribution of zero-mean gridded samples to determine how to scale range of data to -1 to 1.

        Args:
            self: Gridder object
            
        Returns:
            1D vector containing all zero-mean distance values from every grid sample
            
        """
        
        # define the height and width of the fully loaded map
        height = self.full_img.shape[0]
        width = self.full_img.shape[1]

        # determine the step size based on the window size and overlap %
        step_size = self.window_size - int(np.floor(self.window_size*self.overlap))

        # initialize iteration variables
        r_idx = 0
        c_idx = 0
        i = 0
        out = []

        # begin the main loop
        while r_idx + self.window_size <= height:
            while c_idx + self.window_size <= width:
                # define the grid
                box = self.full_img[r_idx:(r_idx+self.window_size),c_idx:(c_idx+self.window_size)]

                # filter out data that contains gaps
                if ~np.any(box.mask): # if there are not any masked entries
                    # zero-mean the data
                    box_zm = (box - np.median(box)).astype(np.float32)

                    # append data to 1D vector containing all image values
                    out.append(box_zm.reshape(-1))
                
                    # increment the data counter
                    i += 1

                # increment the column index
                c_idx += step_size

            # increment the row index, reset the column index back to the beginning
            r_idx += step_size
            c_idx = 0
        
        return np.concatenate(out), i