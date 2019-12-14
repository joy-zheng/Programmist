import sys
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import preprocess
import dgl

class Discriminator_Model(nn.Module):
    def __init__(self):
        super(Discriminator_Model, self).__init__()
        """
        The model for the discriminator network is defined here.
        """

        self.learning_rate = 0.001

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=(1,1))
        self.conv2 = nn.Conv2d(69, 128, kernel_size=4, stride=2)
        self.batchnorm2 = nn.BatchNorm2d(128, eps=0.001, track_running_stats=True)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2)
        self.batchnorm3 = nn.BatchNorm2d(256, eps=0.001, track_running_stats=True)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2)
        self.batchnorm4 = nn.BatchNorm2d(512, eps=0.001, track_running_stats=True)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=4, stride=2)
        self.beta1 = 0.5
        self.beta2 = 0.99
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate, betas=(self.beta1, self.beta2))
        self.mse = nn.MSELoss()



    def forward(self, inputs, labels):
        """
        Executes the discriminator model on a batch of input images and outputs whether it is real or fake.

        :param inputs: a batch of images, shape=[batch_size, height, width, channels]

        :return: a batch of values indicating whether the image is real or fake, shape=[batch_size, 1]
        """
        x = self.leakyrelu(self.conv1(inputs))
        labels = labels[:, :, 0:64, 0:64]
        x = torch.cat((x,labels),1)
        x = self.leakyrelu(self.batchnorm2(self.conv2(x)))
        x = self.leakyrelu(self.batchnorm3(self.conv3(x)))
        x = self.leakyrelu(self.batchnorm4(self.conv4(x)))
        x = self.conv5(x)
        return x

    def loss_function(self, d_real_real, d_fake1_true, d_fake2_false):
        """
        Outputs the discriminator loss given the discriminator model output on the real and generated images.

        :param disc_real_output: discriminator output on the real images, shape=[batch_size, 1]
        :param disc_fake_output: discriminator output on the generated images, shape=[batch_size, 1]

        :return: loss, the combined cross entropy loss, scalar
        """
      # TODO: Calculate the loss
        # d_loss_real = torch.mean(torch.pow((d_real_real - 1.), 2))
        # d_loss_fake1 = torch.mean(torch.pow(d_fake1_true, 2))
        # d_loss_fake2 = torch.mean(torch.pow(d_fake2_false, 2))
        d_loss_real = self.mse(d_real_real, torch.ones(d_real_real.size()).cuda())
        d_loss_fake1 = self.mse(d_fake1_true, torch.zeros(d_real_real.size()).cuda())
        d_loss_fake2 = self.mse(d_fake2_false, torch.zeros(d_real_real.size()).cuda())
        # d_loss_real = self.mse(d_real_real, torch.ones(d_real_real.size()))
        # d_loss_fake1 = self.mse(d_fake1_true, torch.zeros(d_real_real.size()))
        # d_loss_fake2 = self.mse(d_fake2_false, torch.zeros(d_real_real.size()))
        loss = 1./2* (d_loss_real + 1./2*(d_loss_fake1 + d_loss_fake2))

        print("Discriminator loss:", loss)
        return loss
