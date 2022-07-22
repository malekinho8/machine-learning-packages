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


# ### Gridder Class
#
# The goal for this class is to allow for various methods such as creating a gridded data set and saving array data to a folder, determining the distribution of zero-mean data samples, and potentially converting data to an image format that could be saved in an image folder, along with the transformations that allow the user to return the data to the true range.

class Gridder():
    def __init__(self, full_img, window_size, overlap, im_dir, prefix=None):
        """
        The Gridder class, which contains various methods including the ability to create a gridded data set,
        plot a histogram of sample values, and more...

        Args:
            full_img: masked np array of shape (height,width) corresponding to fully loaded bathymetry data
            window_size: int, dimension of the output grids
            overlap: float in the range [0, 1), how much the grids should overlap with one another
            im_dir: str, path to save images
            prefix: (optional) str, what to include when saving each file            

        """
        
        self.full_img = full_img
        self.window_size = window_size
        self.overlap = overlap
        self.im_dir = im_dir
        self.prefix = prefix

    def grid_zm_scaled(self, scale_factor):
        """
        Saves -1 to 1 scaled zero-mean gridded data set as numpy arrays from a bathymetry map.

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
                box = full_img[r_idx:(r_idx+self.window_size),c_idx:(c_idx+self.window_size)]
                
                # filter out data that contains gaps
                if ~np.any(box.mask): # if there are not any masked entries
                    if not os.path.exists(os.path.join(self.im_dir, f'{self.prefix}_{i:06d}.npy')):
                        # zero-mean the data
                        box_zm = (box - np.median(box)).astype(np.float32)
                        
                        # save the data
                        np.save(os.path.join(self.im_dir, f'{self.prefix}_{i:06d}.npy'), box_zm) 
                        i += 1

                # increment the column index
                c_idx += step_size

            # increment the row index
            r_idx += step_size

        print(f'Total Training Samples: {i}')
        
    def histogram_zm(self):
        """
        Creates a distribution of zero-mean gridded samples to determine how to scale range of data to -1 to 1.

        Args:
            self: Gridder object
            
        Returns:
            1D vector containing all zero-mean distance values from every grid sample.
            
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
                    i += 1

                # increment the column index
                c_idx += step_size

            # increment the row index
            r_idx += step_size
        
        return np.concatenate(out)
