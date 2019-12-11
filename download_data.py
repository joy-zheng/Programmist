from __future__ import print_function
import os
import requests
from tqdm import tqdm
import tarfile

def download_file(destination):
    URL = "http://www.umiacs.umd.edu/~sirius/CACD/celebrity2000_meta.mat" 
    response = requests.get(URL, allow_redirects=True)

    session = requests.Session()
 
    save_response_content(response, destination)

def get_confirm_token(response):
  for key, value in response.cookies.items():
    if key.startswith('download_warning'):
      return value
  return None

def save_response_content(response, destination, chunk_size=32*1024):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size), total=total_size, unit='B', unit_scale=True, desc=destination):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_celeb_meta(dirpath):
    filename  = "celebrity2000_meta.mat" 

    save_path = os.path.join(dirpath, filename)

    if os.path.exists(save_path):
        print('[*] {} already exists'.format(save_path))
    else:
        download_file(save_path)

def download_face_images(dirpath):
    filename, drive_id  =  "CACD2000.tar.gz" , "0B3zF40otoXI3OTR0Y0MtNnVhNFU"
    save_path = os.path.join(dirpath, filename)

    if os.path.exists(save_path):
        print('[*] {} already exists'.format(save_path))
    else:
        download_file_from_google_drive(drive_id, save_path)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={ 'id': id }, stream=True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def prepare_data_dir(path = './data'):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    prepare_data_dir()
    download_celeb_meta('./data')
    download_face_images('./data')
    tf = tarfile.open("./data/CACD2000.tar.gz")
    tf.extractall()
