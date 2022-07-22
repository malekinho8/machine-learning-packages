import argparse
import traceback
import shutil
import logging
import yaml
import sys
import os
import torch
import numpy as np
import torch.utils.tensorboard as tb

from runners.diffusion_mmi_good import Diffusion

torch.set_printoptions(sci_mode=False)

# DEFINE SETTINGS HERE
class config:
    class args:
        ni = True # optional; controls whether you want to overwrite files or not (default True)
        timesteps = 20 # number of reverse time steps used in the DDRM model (default 20)
        eta = 0.85 # this is related to the decay rate I believe (default 0.85)
        etaB = 1 # also related to the decay rate (default 1)
        deg = 'sr8' # required; the type of degradation that is desired to be reversed
        sigma_0 = 0 # optional; if you would also like to denoise the input
        log_path = None # directory/folder for storing data logs (optional)
        data_dir = 'sandbox\\datasets' # required; directory containing datasets
        data_folder = 'MBARI-Monterey-Canyon_25-m_low-resolution_2570x64x64' # required; folder name containing images
        model_dir = 'sandbox\\models' # required; directory containing model folders
        model_folder = 'ddpm-MBARI-Monterey-Canyon-5-m_64x64' # required; name of the folder containing the saved model of interest
        out_dir = 'sandbox\\outputs' # required; this is where batches of images are stored after every sampling iteration. This folder is mainly for diagnostic purposes and if the user would like to compare DDRM results on several samples at a time (or compose a GIF showing the reverse process).
        image_folder = 'test_gdrive_lowres_1c' # optional; if name is 'default', it will save images to a folder name in data_dir that will be automatically determined from other settings in config
        comment = '' # optional; if you want to add a comment to experiment (default empty)
        verbose = 'info' # info | debug | warning | critical (default info)
        sample = True # whether to store samples from model (default True)
        subset_start = -1 # default -1, set to a value > 0 if you want to only deal with a slice of the data set
        subset_end = -1 # " "
        seed = 1234

    class data:
        dataset = "USGS_monterey_bay"
        category = "bathymetry"
        image_size = 64
        channels = 1
        logit_transform = False
        uniform_dequantization = False
        gaussian_dequantization = False
        random_flip = True
        rescaled = True
        num_workers = 0
        out_of_dist = False

    class model:
        type = "ddpm_ho"
        file_name = 'model-76.pt' # the file name of the saved model 
        in_channels = 1
        out_ch = 1
        ch = 64
        ch_mult = [1, 2, 4, 8]
        num_res_blocks = 8
        attn_resolutions = [16, ]
        dropout = 0.0
        var_type = 'fixedsmall'
        ema_rate = 0.995
        ema = True
        resamp_with_conv = True

    class diffusion:
        beta_schedule = 'cosine'
        beta_start = 0.0001
        beta_end = 0.02
        num_diffusion_timesteps = 1000

    # relates to settings designated to train the model; not entirely sure 
    class training:
        batch_size = 32
        n_epochs = 100
        n_iters = 100000
        snapshot_freq = 1000
        validation_freq = 1000

    # relates to optimizer settings of the trained network. Used to load the diffusion model object.
    class optim:
        weight_decay = 0.000 # depends on weight decay value used in creation of model; for this model, weight decay was set to 0
        optimizer = "Adam" # " "
        lr = 0.00002 # " "
        beta1 = 0.9 # " "
        amsgrad = False # " "
        eps = 0.00000001 # " "

    # relates to desired settings for producing DDRM samples
    class sampling:
        batch_size = 16 # the number of data instances to restore at a time
        last_only = True # default True
    
# create the logging path directory if it doesn't exist
if config.args.log_path is None:
    config.args.log_path = os.path.join(config.args.out_dir,f'logs_{config.args.data_folder}_{config.args.deg}_{config.model.type}')

    if not os.path.exists(config.args.log_path):
        os.mkdir(config.args.log_path)

# create image folder for output samples
if config.args.image_folder is None:   
    config.args.image_folder = os.path.join(config.args.out_dir, f'{config.args.data_folder}-{config.args.deg}')
else:
    config.args.image_folder = os.path.join(config.args.out_dir, config.args.image_folder)

    if not os.path.exists(config.args.image_folder):
        os.mkdir(config.args.image_folder)



def main():
    logging.info("Writing log file to {}".format(config.args.log_path))
    logging.info("Exp instance id = {}".format(os.getpid()))
    logging.info("Exp comment = {}".format(config.args.comment))

    runner = Diffusion(config)
    runner.sample()
        

    return 0


main()
