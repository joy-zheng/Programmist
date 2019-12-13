import numpy as np
import cv2
import os  
from scipy.io import loadmat
from random import randint 
import json


metadata_dir = 'data/celebrity2000_meta.mat'
image_dir  = 'data/CACD2000' 
age_groups = [range(11, 21), range(21, 31), range(31, 41),range(41, 51), range(51, 151)]

def get_batch_metadata():
    """
    Gets the metadata from metadata directory and returns the metadata for both celebrities and images.

        :param metadata_dir: directory of metadata

        :return: two arrays of celebrity and image metadata
    """ 
    x = loadmat(metadata_dir)
    datatype = ['celebrityData', 'celebrityImageData']
    # names = x[datatype[0]][0][0][0]
    # identity = x[datatype[0]][0][0][1]
    # birth = x[datatype[0]][0][0][2]
    # rank = x[datatype[0]][0][0][3] #rank of the celebrity with same birth year in IMDB.com when the dataset was constructed
    # lfw = x[datatype[0]][0][0][4] #whether the celebrity is in LFW dataset 
    # celeb_metadata = [datatype, names, identity, birth, rank, lfw] #array of celeb features

    paths = np.asarray(sorted(os.listdir(image_dir)))
    image_age = x[datatype[1]][0][0][0] 

    image_age = np.asarray(sorted(image_age))
    age_group_labels = [0 if (i in age_groups[0]) else 1 if (i in age_groups[1]) else 2 if (i in age_groups[2]) else 3 if (i in age_groups[3]) else 4 if (i in age_groups[4]) else None for i in image_age] 
    # image_metadata = [image_age, age_group_labels, image_id, image_year, image_features, image_filename] #array of image features

    paths = np.reshape(paths, (paths.shape[0], 1))
    age_group_labels = np.reshape(age_group_labels, (len(age_group_labels), 1))
    ages_path = np.concatenate([paths, age_group_labels], axis = 1)   
    # np.random.shuffle(agesd_path) 
    return  ages_path

array = get_batch_metadata()

with open('ages_paths.txt', 'w') as filehandle:
    json.dump(array.tolist(), filehandle)