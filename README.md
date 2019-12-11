# Programmist
Final Project for CSCI 1470 Deep Learning - Face-Aging with Identity Preserved Conditional GANs

We are implementing the Face Aging With Identity-Preserved Conditional Generative Adversarial Networks paper by Xu Tang, Zongwei Wang, Weixin Luo and Shenghua Gao. We will use GANs to extract low-level features from faces of different ages and encode the features to our images. This is a generative model with a face-aging objective. 

Face-aging is a new and exciting field in Deep Learning. With the rising popularity of FaceApp and people’s concerns about privacy, we were drawn to implement our version of an aging-app and think over issues of data privacy, deep fakes and identity theft.
 
## How to Train (Cross-Age Celebrity Dataset)
1. Run `python download_data.py` to download the the [Cross-Age Celebrity Dataset metadata](http://www.umiacs.umd.edu/~sirius/CACD/celebrity2000_meta.mat)(817K) and [Face images](https://drive.google.com/file/d/0B3zF40otoXI3OTR0Y0MtNnVhNFU/)(3.5G). These will be downloaded and unzipped in the `data\` directory. 

## Environment

#### Navigating GCP

```
python
from zipfile import ZipFile
zip_file = ZipFile('path_to_file/t.zip', 'r')
zip_file.extractall('path_to_extract_folder')

```

#### Setup conda

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

bash Miniconda3-latest-Linux-x86_64.sh

rm Miniconda3-latest-Linux-x86_64.sh

source .bashrc

```

#### Installation

`conda env create -f environment_gpu.yml`

#### Update

`conda env update -f environment_gpu.yml`

#### Activate/Deactivate

`conda activate clsgan`

`conda deactivate`
