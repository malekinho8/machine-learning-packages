# # **DDPM Super-Resolution Evaluation**
#
# # By Malek Ibrahim, 07.19.2022

# The goal of this code is to develop some framework which can load some validation/evaluation data from a specified directory iteratively, degrade it, and ultimately obtain metrics quantifying SR performance (PSNR, SSIM, etc.).

# +
# Proprietary and Confidential
# Copyright 2003 - 2020 Stone Aerospace
# 3511 Caldwell Lane, Del Valle, TX 78617
# All rights reserved
# -

# import necessary packages
import numpy as np
from rasterio.plot import show
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 8]
import cv2
import os
from bathymetry_utils.analysis import plot_im_list, psnr, ssim, quantize, listdir_fullpath
import torch
import sys
from skimage.measure import block_reduce
import math
from datetime import datetime
from tqdm.auto import tqdm
import pandas as pd
from ddrm_codes_2.functions.denoising import efficient_generalized_steps
from ddrm_codes_2.runners.diffusion_mmi_good import get_beta_schedule
import matplotlib.patches as patches
from torchvision import transforms
from ddrm_codes_2.functions.svd_replacement import SuperResolution

def load_sample(path):
        """ Loads a data sample from the path specified.

        Args:
            path: str, data file containing image to be evaluated 

        Returns:
            loaded: ndarray, loaded data sample, quantized to [0,255]

        """
        if '.npy' in path:
            loaded = np.load(path)
        elif '.jpg' in path or '.png' in path:
            loaded.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

        return loaded

class SR_Evaluator():
    """ The goal of this code is to develop some framework which can load some validation/evaluation data from a specified directory iteratively, 
    degrade it, and ultimately obtain metrics quantifying SR performance (PSNR, SSIM, etc.).
    """
    def __init__(self, path_to_samples, sr_factor):
        self.path = path_to_samples
        self.r = sr_factor
        self.H = None
        self.device = None
        self.model = None
        self.model_path = None

    def degrade(self, sample):
        return block_reduce(sample, block_size=self.r, func=np.mean)

    def super_resolve(self, y_0, mode, ddrm_timesteps = 20, etaB = 1, eta = 1, sigma_0 = 0, cls_fn = None, classes = None):
        """ This function takes in a low-res sample and then super-resolves it according to the method specifed by mode.

        Args:
            y_0: 2D array, the low resolution sample
            mode: one of either 'ddrm', 'bicubic', or 'psuedo-inverse'
            **ddrm_kwargs

        Returns:
            out: 2D array same shape as high resolution sample
        """
        if mode == 'ddrm':
            if self.H is None:

                if 'win' in sys.platform:
                    self.model_path = str(input('\nPlease paste the path to the model below: \n')).replace('\\',os.sep).replace('"','')
                else:
                    self.model_path = str(input('\nPlease paste the path to the model below: \n'))
                
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # load and set up the model
                mdata = torch.load(self.model_path, map_location=self.device)
                self.model = mdata['diffusion_model']
                self.model_denoise = self.model.denoise_fn
                self.model_denoise.to(self.device)
                self.model_denoise.eval()
                self.model_denoise = torch.nn.DataParallel(self.model_denoise)

                assert self.model.image_size == y_0.shape[0]*self.r, 'Sample must have same output/input size to diffusion model.'
                self.H = SuperResolution(self.model.channels, self.model.image_size, self.r, self.device)

            # specify ddrm variables
            y_0 = torch.tensor(y_0).view(1,-1).to(self.device)
            x = torch.randn(1, self.model.channels, self.model.image_size, self.model.image_size, device=self.device) # initialize noise
            num_diffusion_timesteps = self.model.num_timesteps
            skip = num_diffusion_timesteps // ddrm_timesteps
            seq = range(0, num_diffusion_timesteps, skip)
            pred_noise = True if self.model.objective=='pred_noise' else False

            ddrm_out = efficient_generalized_steps(x, seq, self.model_denoise, pred_noise, self.model.betas, self.H, y_0, sigma_0, \
            etaB=etaB, etaA=eta, etaC=eta, cls_fn=cls_fn, classes=classes)
            out = ddrm_out[0][-1][0,0]

        if mode == 'bicubic':
            out = cv2.resize(y_0, (y_0.shape[0]*self.r,y_0.shape[1]*self.r), cv2.INTER_CUBIC)
        
        if mode == 'pseudo-inverse':
            out = y_0.repeat(self.r, axis=0).repeat(self.r, axis=1)

        return out

    def validation_metrics(self, save_flag=True, mode_list=['ddrm', 'bicubic', 'pseudo-inverse']):
        """ Loops over whole validation dataset folder and appends various metrics to a dictionary output according to the specified SR mode.
        
        Args:
            mode_list: List containing one or several of 'ddrm', 'bicubic', or 'psuedo-inverse' to loop over.

        Returns:
            out: dict of metrics calculated between the SR image and the original image on all validation samples
        """
        if save_flag:
            save_folder = str(input('\n Where would you like to save the metrics for this run (Paste absolute or relative path to folder)?\n')).replace('"','')
            self.metrics_folder = save_folder
            self.time_tag = time_tag
            time_tag = str(datetime.now()).split(' ')[0]
            self.save_name = f'{self.r}x_Test_{time_tag}'

        files = listdir_fullpath(self.path)
        out = {x: {'PSNR': [], 'SSIM': [], 'LR-PSNR': []} for x in mode_list} # initialize the output
        
        # num_batches = len(files)//batch_size
        # full_range = np.arange(len(files))
        # batches = [slices[]]

        for i in tqdm(range(len(files))):
            # batch_slice = full_range[i*batch_size:(i+1)*batch_size]
            for mode in mode_list:
                # perform the main flow: load sample --> degrade sample --> super-resolve --> calculate metrics --> output to dict
                f = files[i]

                if '.png' in f or '.npy' in f or '.jpg' in f:
                    img = load_sample(f) # load the sample
                    y_0 = self.degrade(img) # degrade the sample according to specified SR factor
                    sr_out = self.super_resolve(y_0, mode) # obtain the high-resolution output
                    LR_out = self.degrade(sr_out) # downsample the SR output to calculate LR-PSNR --> Lugmayr et. al 2020

                    # calculate the metrics
                    PSNR = psnr(sr_out, img)
                    SSIM = ssim(sr_out, img)
                    LR_PSNR = psnr(LR_out, y_0)

                    out[mode]['PSNR'].append(PSNR)
                    out[mode]['SSIM'].append(SSIM)
                    out[mode]['LR-PSNR'].append(LR_PSNR)

        if save_flag:
            torch.save(out, f'{self.metrics_folder}{os.sep}{self.save_name}')
            mean_std = self.get_mean_std(out)

        return {
            'Raw Validation Metrics': out,
            'Mean and Std': mean_std
            }

    def get_mean_std(self, raw_metrics):
        """ This method computes the mean and standard deviation of a raw metrics data structure which contains all of the raw validation metrics.

        Args:
            raw_metrics: dict with structure mode --> metric name over which mean and std will be calculated

        Returns:
            out: dict with same structure, reduced to mean and standard deviation for each entry

        """
        
        out_name = 'mean_std_metrics'
        mode_list = list(raw_metrics.keys())
        metric_list = list(raw_metrics[mode_list[0]].keys())
        temp = {x: {y: {} for y in metric_list} for x in mode_list}

        for mode in mode_list:
            for metric in metric_list:
                mean = np.nanmean(raw_metrics[mode][metric])
                std = np.nanstd(raw_metrics[mode][metric])
                temp[mode][metric]['mean'] = mean
                temp[mode][metric]['std'] = std
        
        save_out = os.path.join(self.metrics_folder,out_name)
        torch.save(temp, save_out)

        return out
                
            
# print('Done Loading')
# A = SR_Evaluator(r"I:\My Drive\sandbox\datasets\MBARI-Monterey-Canyon_5-m_hi-res_npy_13937x60x60 (1d)\validation\0", 10)
# A.validation_metrics()
# print('test')