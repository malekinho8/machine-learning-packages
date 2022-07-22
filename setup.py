from setuptools import setup

setup(
    name="custom-ml-packages",
    py_modules=["bathymetry_utils", "ddrm_codes_2", "denoising_diffusion_pytorch", "einops", "torchvision_mmi"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)


