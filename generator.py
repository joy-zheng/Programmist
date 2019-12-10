import sys
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import preprocess
import dgl
import torchvision.models as models
import numpy as np
import cv2
import skimage
from PIL import Image
from torchvision import transforms
 

class Generator_Model(nn.Module):
    def __init__(self):
        """
        The model for the generator network is defined here.
        """
        super(Generator_Model, self).__init__()
        # TODO: Define the model, loss, and optimizer
        
        # Initialize hyperparameters
        self.learning_rate = 5e-4
        self.batch_size = 10
        self.epochs = 15
        self.generator_weight = 0.4
        self.age_weight = 0.3
        self.identity_weight = 0.3
    
        # Initialize layers
        self.conv1 = nn.Conv2d(8, 32, kernel_size=7, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32, eps=0.001)
        self.bn2 = nn.BatchNorm2d(64, eps=0.001)
        self.bn3 = nn.BatchNorm2d(64, eps=0.001)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=7, stride=1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        self.tanh = nn.Tanh()

    def forward(self, inputs, labels):
        """
        Executes the generator model on the random noise vectors.
        :param inputs: images and conditional feature maps concatenated together.
        :return: prescaled generated images, shape=[batch_size, height, width, channel]
        """
        # TODO: Call the forward pass

        # x = torch.cat((torch.from_numpy(inputs),torch.from_numpy(labels)),3)
        print(inputs.shape)
        print(labels.shape)
        x = np.concatenate((inputs, labels), 1)
        x = torch.tensor(x).float()
        # x = x.permute(0, 1, 3, 2).float()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        print("Conv 1 Shape:", x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        print("Conv 2 Shape:", x.shape)
        x = self.bn3(x)
        x = self.conv3(x)
        x = self.relu(x)
        print("Conv 3 Shape:", x.shape)
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.relu(x)

        x = self.conv4(x)
        print("generator output is ", x.shape)
        x = self.tanh(x)
        return x

    def loss_function(self, real_img, fake_img, condition):
        """
        Outputs the loss given the discriminator output on the generated images.
        :param disc_fake_output: the discrimator output on the generated images, 
        :return: loss, the cross entropy loss, scalar
        """
        # TODO: Calculate the loss
        # fake_img = self.call(real_img, target_ae_group)
        generator_loss = (1 / 2 * np.mean(np.square(fake_img - 1)))
        age_loss = self.calculate_age_loss(fake_img, condition)
        identity_loss = self.identity_preserving_module(real_img, fake_img)
        weighted_loss = self.generator_weight * generator_loss + self.age_weight * age_loss + self.identity_weight * identity_loss
        return weighted_loss
    
    def calculate_age_loss(self, fake_img, target_age_group):
        """
        Calculate age loss for generator
        :param batched_real_img: a batch of training images, shape=[batch_size, img_height, img_width, num_channel]
        :param target_age_grouup: a list of target age groups, shape=[num_age_group]
        :return a float loss for the batch
        """
        fake_age = self.classify_age_alexnet(fake_img)
        age_loss = self.softmax_cross_entropy_loss(fake_age, target_age_group)
        #    age_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #                logits=fake_age, labels=target_age_group))
        return age_loss

    def classify_age_alexnet(self, fake_img):
        """
        Use a pre-trained alexnet age classifier to label the fake images with age groups
        :param fake_img: a set of fake images generated from the batch, shape=[batch_size, img_height, img_width, num_channel]
        :return fake_age, shape=[batch_size, num_age_group]
        """
        # https://pytorch.org/hub/pytorch_vision_alexnet/ for reference
        num_age_group = 5
        alexnet = models.alexnet(pretrained=True)
        alexnet.classifier[6] = nn.Linear(in_features=4096, out_features=num_age_group)
        alexnet.eval()
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # assuming input is numpy array
        fake_img = np.swapaxes(np.swapaxes(fake_img, 1,3), 2, 3)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        processed_img = []
        for img in fake_img:
            sample = preprocess(img)
        processed_img = torch.Tensor(processed_img)
        output = alexnet(processed_img)
        _, pred = np.max(output,1)
        return pred

    def softmax_cross_entropy_loss(self, logits, labels):
        """
        Calculate softmax cross entropy loss from given logits and labels
        
        :param logits: fake_age, shape=[batch_size, num_age_group]
        :param labels: target_age_grouup, shape=[num_age_group]
        :return a float loss for the batch
        """
        # checkout https://discuss.pytorch.org/t/pytorch-equivalence-to-sparse-softmax-cross-entropy-with-logits-in-tensorflow/18727/2
        loss = nn.CrossEntropyLoss()
        output = loss(logits, labels)
        return np.mean(output)


    def identity_preserving_module(self, org_image, generated_image):
        """
        Outputs the identity loss for the given image and its generated image.
        :param disc_fake_output: original_features and generated_features from the 8th layer of AlexNet.
        :return: the sum of mean squared identity loss
        """
        original_features = self.alex_features(org_image)
        generated_features = self.alex_features(generated_image)
        identity_loss =  0
        for i in len(original_features):
            sq_diff = np.square(abs(original_features[i] - generated_features[i]))
            identity_loss +=  sq_diff        
        return identity_loss

    def alex_features (self, input_image):
        # input_image = Image.open('bunny.jpg')
        alexnet_model = models.alexnet(pretrained=True)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = alexnet_model(input_batch)
        return output