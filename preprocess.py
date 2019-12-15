import numpy as np
import cv2
import os  
from scipy.io import loadmat
from random import randint 
import json 

class Data_Processor:
    def __init__(self, batch_size = 100, image_size= 128, shuffle=True, mode='train'):
        self.metadata_dir = 'data/celebrity2000_meta.mat'
        self.image_dir  = 'data/CACD2000' 
        self.image_size  = image_size 
        self.batch_size = batch_size 
        self.train_pointer = 0 
        self.test_pointer = 0
        self.mode = mode
 

    def get_next_batch_image(self):
        """
        Gets the image from  image path and returns the image.
            Given an image data directory, this function opens and decodes the image stored in the directory.

            :param image_dir: directory of image data

            :return: arrays  of rgb images and paths
        """ 
        ages_path = np.array(json.loads(open("ages_paths.txt","r").read())) #load  from  json file.
        dataset_size = len(ages_path) 
        if self.mode == 'train':
            n = self.batch_size*self.train_pointer 
        if self.mode == 'test':
            n = self.batch_size*self.test_pointer + int(dataset_size*0.9)  
        
        paths  = ages_path[n:n+self.batch_size,0] 
        imgs = np.ndarray([len(paths),  self.image_size, self.image_size, 3], dtype= np.float32)
        for i in range(len(paths)):
            img = cv2.imread(os.path.join(self.image_dir, paths[i]))
            if len(np.asarray(img).shape) > 0 :  
                img = cv2.resize(img, (self.image_size, self.image_size)) 
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
                
                imgs[i] = img  
        imgs = (imgs/127.5)-1
        imgs = np.swapaxes(imgs, 1, 3)
         

        real_labels = np.asarray(ages_path[n:n+self.batch_size,1], dtype = int) 
 
        train_label_pairs = self.get_fakelabels(real_labels)   #Generate [real_label, fake_label] pairs
        # train_label_pairs = self.get_targetfakelabels(real_labels) #Generate [target_label, fake_label] pairs
        
        fake_labels =  train_label_pairs[:,1]
        real_labels_onehot = np.zeros((len(paths), 5, self.image_size, self.image_size))
        real_labels_onehot[np.arange(len(paths)), real_labels, :,:] = np.ones((self.image_size,self.image_size)) 
        fake_labels_onehot = np.zeros((len(paths), 5, self.image_size, self.image_size))
        fake_labels_onehot[np.arange(len(paths)), fake_labels, :,:] = np.ones((self.image_size,self.image_size))  
        if self.mode == 'train':
            self.train_pointer += 1  
        if self.mode == 'test':
            self.test_pointer += 1 
        return imgs, real_labels_onehot, fake_labels_onehot, train_label_pairs, paths

    def get_fakelabels(self, true_labels):
        label_pairs = np.zeros((len(true_labels),2), dtype=int)
        label_pairs[:,0] = true_labels
        n = max(true_labels)
        for i in range(len(true_labels)):
            rand = randint(1,n)
            true_label = true_labels[i]
            fake_label = (true_label+rand)%n
            label_pairs[i,1] = fake_label 
        return label_pairs

    def get_targetfakelabels(self, true_labels):
        label_pairs = np.zeros((len(true_labels),2), dtype=int) 
        n = max(true_labels)
        for i in range(len(true_labels)):
            rand = randint(1,n)
            true_label = true_labels[i]
            target_label = (true_label+rand)%n
            label_pairs[i,0] = target_label
        target_labels = label_pairs[:,0] 
        for i in range(len(true_labels)):
            rand = randint(1,n)
            true_label = target_labels[i]
            fake_label = (true_label+rand)%n
            label_pairs[i,1] = fake_label
        return label_pairs
    