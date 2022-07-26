# -*- coding: utf-8 -*-
"""ddpm_csv_analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
        https://colab.research.google.com/drive/1F0o63bWnJettOvWYmHyayQZpBF7XmUl1

# **CSV Analysis Tools**

# By Malek Ibrahim, 07.04.2022

The goal of this code is to develop various functions to extract data from csv files saved during training of DDPMs. Such files currently include validation loss over time, training loss over time, and DDRM PSNR on validation set mean + standard deviation.
"""

# Proprietary and Confidential
# Copyright 2003 - 2020 Stone Aerospace
# 3511 Caldwell Lane, Del Valle, TX 78617
# All rights reserved

# import modules/dependencies
import matplotlib
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch
import cv2
from scipy import stats
from datetime import datetime
plt.rcParams['figure.figsize'] = [16, 8]

def quantize(ndarray):
    """ This function converts an ndarray scaled between two arbitrary numbers to the range [0,255]

    Args:
        ndarray: A torch or numpy array object

    Returns:
        A numpy array with the same shape as ndarray, scaled between [-1,1]

    """
    temp = ndarray - ndarray.min() # scale to [0, N]
    temp = temp/temp.max() # scale to [0, 1]

    # currently only adapted to handle torch.Tensors and numpy arrays
    if isinstance(temp, torch.Tensor):
        out = (temp*255).to(int).cpu().numpy()
    elif isinstance(temp, np.ndarray):
        out = (temp*255).astype(int)

    return out

def ssim(img1, img2):
    """ Takes two images as inputs. If the data is not scaled between 0 and 255, a warning message will come up and scaling will occur. 
    
    Args:
        img1: a 2D array
        img2: a 2D array
    
    Returns:
        The SSIM between the two images
    """

    # quantize the images if necessary
    if img1.max() != 255 or img1.min() != 0:
        img1 = quantize(img1)
    
    if img2.max() != 255 or img2.min() != 0:
        img2 = quantize(img2)

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def psnr(img1, img2):
    # quantize the images if necessary
    if img1.max() != 255 or img1.min() != 0:
        img1 = quantize(img1)
    
    if img2.max() != 255 or img2.min() != 0:
        img2 = quantize(img2)

    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def running_mean(data, window_size, interpolate_flag=True, window_step=None):
    """Computes running mean of a data array.
                
    Args:
        data: A 1D array, or str corresponding to csv data file name
        window_size: how many values to average at a time
        interpolate_flag: boolean, default True; if you want the data output to be interpolated instead of quantized
        window_step: optional, how much to step over to take the mean of the next window; default = window_size
            
    Returns:
        The running mean of the data input, same shape as data if interpolate_flag is False, less values if interpolate_flag is True.
    """
    if window_step is None:
        window_step = window_size

    if isinstance(data,str):
        data = np.array(pd.read_csv(os.path.join(self.out_dir,data)).iloc[:,-1])
    
    # initialize the running mean, iteration counter
    output = np.zeros_like(data)
    length = 0
    count = 0

    assert len(data) >= window_size, 'The window size over which the mean is computed must be smaller than the data length. Please reduce the window size and try again.'

    while len(data) - length >= window_size:
        output[window_step*count:window_step*(count+1)] = np.mean(data[window_step*count:window_size+window_step*count])
        length += window_step
        count += 1
    
    # take mean of remained of data array and set equal to remaining slice of running_mean
    output[window_step*(count-1)::] = np.mean(data[window_step*(count-1)::])

    if interpolate_flag:
        out = output[0::window_step]

    return out

def plot_im_list(im_list, n_samples=None, show_flag=False, **imshowargs):
    """ Plots a sqaure grid of samples
    
    Args:
        im_list: A list containing image samples, either in the form of paths to .png's, or .npy 2D array files.
        n_samples: default None; if an integer, the plot will only contain as many samples as that specified here
        show_flag: whether or not to show the plot or simply return the figure
        imshowargs: any additional arguments to pass to the imshow, such as vmin or vmax (to set the range of the data)

    Returns:
        fig: A matplotlib.pyplot figure object which contains plotted data samples

    """
    if isinstance(im_list[0], str):
        assert n_samples != None, 'n_samples must be specified since the list given contains paths to image files.'
        
        # generate a random index vector of length n_samples
        vec = np.arange(0,len(im_list))
        choice = np.random.choice(vec, n_samples, replace=False)
        im_list = np.array(im_list)[choice].tolist()

        if '.png' in im_list[0] or 'jpg' in im_list[0] or 'jpeg' in im_list[0]:
            assert os.path.exists(im_list[0]), 'Path does not exist...'
            im_list = [cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in im_list]
        elif '.npy' in im_list[0]:
            im_list = [np.load(x) for x in im_list]
        

    # define number of samples
    if n_samples is None:
        n_samples = len(im_list)

    

    rows = cols = int(np.ceil(n_samples**0.5))
    fig, axes = plt.subplots(rows,cols,figsize=(8,8)) 

    c = 0
    for i in range(0,rows):
        for j in range(0,cols):
            if c < n_samples:
                im = im_list[c]
                axes[i,j].imshow(im, **imshowargs)
                axes[i,j].set_xticks([])
                axes[i,j].set_yticks([])
            else:
                axes[i,j].set_axis_off()

            c += 1

    fig.subplots_adjust(wspace=0,hspace=0)

    if show_flag:
        plt.show(block=True)   
    
    return fig



class Analyzer():
    def __init__(self, out_dir):
        """The purpose of this class is to allow for easier analysis and processing of results obtained during training of diffusion models. It contains
        various methods such as plotting loss over time, plotting R-PSNR over time, and plotting potential correlations between measurements.

        Args:
            out_dir: where all of the data outputs/measurements are stored (loss, PSNR, etc.)
        """
        self.out_dir = out_dir


    def plot_loss(self, loss_data, labels, fig_num):
        """Plots and iterates over loss data obtained from a model training run.

        Args:
            loss_data: A list of 1D numpy arrays, or str corresponding to data file name
            labels: optional, a list of str objects corresponding to the items in loss_data
            fig_num: which figure number the plot should correspond to
                
        Returns:
            A plot of the data with all loss plots and minimum loss locations labeled.
                
        """
        # initialize figure
        plt.figure(fig_num)

        # loop over the list
        for i, item in enumerate(loss_data):
            # loads the file if it is a string
            if isinstance(item,str):
                data = np.array(pd.read_csv(os.path.join(self.out_dir,item)).iloc[:,-1])
            else:
                data = item
            
            if 'Val' in labels[i] or 'val' in labels[i]:
                self.val_loss = data 

            x = np.arange(0, len(data)) + 1
            plt.plot(x, data, label=labels[i])
            
        plt.title('Loss Plot')
        plt.legend()
        plt.show(block=False)

    def plot_loss_semilog(self, loss_data, labels, fig_num):
        """Semilog (y) plots and iterates over loss data obtained from a model training run.

        Args:
            loss_data: A list of 1D numpy arrays, or str corresponding to data file name
            labels: optional, a list of str objects corresponding to the items in loss_data
            fig_num: which figure number the plot should correspond to
                
        Returns:
            A plot of the data with all loss plots and minimum loss locations labeled.
                
        """
        # initialize figure
        plt.figure(fig_num)

        # loop over the list
        for i, item in enumerate(loss_data):
            # loads the file if it is a string
            if isinstance(item,str):
                data = np.array(pd.read_csv(os.path.join(self.out_dir,item)).iloc[:,-1])
            else:
                data = item

            if 'Val' in labels[i] or 'val' in labels[i]:
                self.val_loss = data 
            
            x = np.arange(0, len(data)) + 1
            plt.plot(x, data, label=labels[i])
            plt.yscale('log')
            
        plt.title('Semilog Loss Plot')
        plt.legend()
        plt.show(block=False)

    def plot_PSNR(self, mean_data, std_data, fig_num):
        """Plots the mean PSNR data obtained during training along with standard deviation
                    
        Args:
            mean_data: A 1D array or name of csv file, each point represents the mean PSNR calculated for a given batch of validation data
            std_data: A 1D array or name of csv file, each point represent the std of the PSNR calculated for a given batch of validation data
            fig_num: int, figure number to plot the data
                
        Returns:
            A plot of the mean and std for the PSNR data.
        """

        # if necessary, load mean and standard deviation data
        if isinstance(mean_data,str):
            mdata = np.array(pd.read_csv(os.path.join(self.out_dir,mean_data)).iloc[:,-1])
        else:
            mdata = mean_data

        if isinstance(std_data,str):
            sdata = np.array(pd.read_csv(os.path.join(self.out_dir,std_data)).iloc[:,-1])
        else:
            sdata = std_data
        
        # output to object
        self.psnr = mdata

        # initialize the figure
        fig = plt.figure(fig_num)

        # plot the mean data
        p = plt.plot(mdata)
        plt.fill_between(np.arange(len(mdata)), mdata-sdata, mdata+sdata, color=p[0].get_color(), alpha=0.25)

        plt.title('DDRM PSNR Over Model Training')

        plt.show(block=False)
        
        return fig


    def plot_correlation(self, X_data, Y_data, fig_num):
        """
        This function seeks to plot the potential correlation between two variables, X and Y.

        Args:
            X_data_path: str or 1D data vector, represents data to be plotted on x-axis
            X_data_path: str or 1D data vector, represents data to be plotted on y-axis
            fig_num: number of figure for matplotlib

        Returns:
            m: float, slope of the correlation
            b: float, y-intercept of the correlation
            R2: coefficient of correlation

            Scatter plot with linear fit.

        """
        # if necessary, load X and Y data
        if isinstance(X_data,str):
            X = np.array(pd.read_csv(os.path.join(self.out_dir,X_data)).iloc[:,-1])
        else:
            X = X_data
        
        if isinstance(Y_data,str):
            Y = np.array(pd.read_csv(os.path.join(self.out_dir,Y_data)).iloc[:,-1])
        else:
            Y = Y_data

        # obtain linear regression statistics
        m, b, R2, p_value, std_err = stats.linregress(X,Y)

        # create plot
        plt.figure(fig_num)
        plt.scatter(X,Y,label='Raw Data')
        plt.plot(X, m*X + b, label=f'Linear Fit, R2 = {R2:.03f}', color='r')
        plt.legend()
        plt.show(block=False)

        return m, b, R2

    def plot_time(self, data, fig_num):
        """Plots and iterates over loss data obtained from a model training run.

        Args:
            data: a 1D numpy array, or str of data file name
            fig_num: which figure number the plot should correspond to
                
        Returns:
            A plot of model training time vs. iteration number
                
        """
        # initialize figure
        plt.figure(fig_num)

        # loads the file if it is a string
        if isinstance(data,str):
            times = [datetime.fromisoformat(x).timestamp() for x in np.array(pd.read_csv(os.path.join(self.out_dir,data)).iloc[:,0])]
            data = np.diff(times)
            temp = np.median(data)
            print(f'Median Seconds per Iteration: {temp:.4f}')

        x = np.arange(0, len(data)) + 1
        plt.plot(x, data)
            
        plt.title('Training Time Plot')
        plt.legend()
        plt.show(block=False)
    
    def running_mean(self, data, window_size, interpolate_flag=True, window_step=None):
        """Computes running mean of a data array.

        Args:
            data: A 1D array, or str corresponding to csv data file name
            window_size: how many values to average at a time
            interpolate_flag: boolean, default True; if you want the data output to be interpolated instead of quantized
            window_step: optional, how much to step over to take the mean of the next window; default = window_size

        Returns:
            The running mean of the data input, same shape as data if interpolate_flag is False, less values if interpolate_flag is True.
        """

        if window_step is None:
            window_step = window_size

        if isinstance(data,str):
            data = np.array(pd.read_csv(os.path.join(self.out_dir,data)).iloc[:,-1])

        # initialize the running mean, iteration counter
        output = np.zeros_like(data)
        length = 0
        count = 0

        assert len(data) >= window_size, 'The window size over which the mean is computed must be smaller than the data length. Please reduce the window size and try again.'

        while len(data) - length >= window_size:
            output[window_step*count:window_step*(count+1)] = np.mean(data[window_step*count:window_size+window_step*count])
            length += window_step
            count += 1

        # take mean of remained of data array and set equal to remaining slice of running_mean
        output[window_step*(count-1)::] = np.mean(data[window_step*(count-1)::])

        if interpolate_flag:
            out = output[0::window_step]

        return out
