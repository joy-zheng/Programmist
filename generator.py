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
	
        # Initialize layers
		self.conv1 = nn.Conv2D(1, 32, kernel_size=7, stride=)
        self.conv2 = nn.Conv2D(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2D(64, 128, kernel_size=3)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(20)
        self.deconv1 = nn.ConvTranspose2d(3, 64, kernel_size=3)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3)
		self.conv4 = nn.Conv2D(3, kernel_size=7)
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)

    def call(self, inputs):
        """
        Executes the generator model on the random noise vectors.

        :param inputs: images and conditional feature maps concatenated together.

        :return: prescaled generated images, shape=[batch_size, height, width, channel]
        """
        # TODO: Call the forward pass
        x = self.conv1(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.conv3(x)
        x = self.relu(x)

        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.relu(x)

        x = self.conv4(x)


        return nn.Tanh(x)

    def loss_function(self, disc_fake_output):
        """
        Outputs the loss given the discriminator output on the generated images.

        :param disc_fake_output: the discrimator output on the generated images, 

        :return: loss, the cross entropy loss, scalar
        """
        # TODO: Calculate the loss
        return (1 / 2 * np.mean(np.square(disc_fake_output - 1)))


    def identity_preserving_module(org_image, generated_image):
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

    def age_classification_module():
        pass


    def alex_features (input_image):
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
