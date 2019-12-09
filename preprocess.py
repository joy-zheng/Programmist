import numpy as np
import cv2
import os  
from scipy.io import loadmat
from random import randint

metadata_dir = 'data/celebrity2000_meta.mat'
image_dir  = 'data/CACD2000' 
img_size  = 128 
age_groups = [range(11, 21), range(21, 31), range(31, 41),range(41, 51), range(51, 151)]

def get_metadata(metadata_dir):
    """
    Gets the metadata from metadata directory and returns the metadata for both celebrities and images.

        :param metadata_dir: directory of metadata

        :return: two arrays of celebrity and image metadata
    """ 
    x = loadmat(metadata_dir) 
    datatype = ['celebrityData', 'celebrityImageData']
    # print(x)
    names = x[datatype[0]][0][0][0]
    identity = x[datatype[0]][0][0][1]
    birth = x[datatype[0]][0][0][2]
    rank = x[datatype[0]][0][0][3] #rank of the celebrity with same birth year in IMDB.com when the dataset was constructed
    lfw = x[datatype[0]][0][0][4] #whether the celebrity is in LFW dataset 
    celeb_metadata = [datatype, names, identity, birth, rank, lfw] #array of celeb features

    image_age = x[datatype[1]][0][0][0]
    image_id = x[datatype[1]][0][0][1]
    image_year = x[datatype[1]][0][0][2]
    image_features = x[datatype[1]][0][0][3]
    image_filename = x[datatype[1]][0][0][4]

    age_group_labels = [0 if (i in age_groups[0]) else 1 if (i in age_groups[1]) else 2 if (i in age_groups[2]) else 3 if (i in age_groups[3]) else 4 if (i in age_groups[4]) else None for i in image_age] 
    image_metadata = [image_age, age_group_labels, image_id, image_year, image_features, image_filename] #array of image features

    return celeb_metadata, image_metadata

def get_image(image_dir):
    """
    Gets the image from  image path and returns the image.
        Given an image data directory, this function opens and decodes the image stored in the directory.

        :param image_dir: directory of image data

        :return: arrays  of rgb images and paths
    """ 
    paths = os.listdir(image_dir)[0:1000]
    imgs = np.ndarray([len(paths), img_size, img_size, 3])
    for i in range(len(paths)):
        img = cv2.imread(os.path.join(image_dir, paths[i]))
        #uncomment below if  you want to display imgs
        # if i == 0:
        #     cv2.imshow('image', img)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows() 
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32) 
        imgs[i] = img 
    celeb_metadata, image_metadata = get_metadata(metadata_dir)
    age_groups = image_metadata[1][0:len(paths)] 
    train_label_pairs = get_fakelabels(age_groups)
    fake_labels =  train_label_pairs[:,1]
    real_labels_onehot = np.zeros((len(paths), img_size,img_size, 5))
    real_labels_onehot[np.arange(len(paths)),:,:, age_groups] = np.ones((img_size,img_size)) 

    fake_labels_onehot = np.zeros((len(paths), img_size,img_size, 5))
    fake_labels_onehot[np.arange(len(paths)),:,:, fake_labels] = np.ones((img_size,img_size)) 
    return imgs, real_labels_onehot, fake_labels_onehot, train_label_pairs, paths

def get_fakelabels(true_labels):
    label_pairs = np.zeros((len(true_labels),2), dtype=int)
    label_pairs[:,0] = true_labels
    n = max(true_labels)
    for i in range(len(true_labels)):
        rand = randint(1,n)
        true_label = true_labels[i]
        fake_label = (true_label+rand)%n
        label_pairs[i,1] = fake_label 
    return label_pairs

# get_image(image_dir)
 #TODO: might write a next_batch function to make batching easier
