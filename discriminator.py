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
        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)
        #
        # self.model = nn.Sequential(
        #     nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 512),
        #     nn.Dropout(0.4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 512),
        #     nn.Dropout(0.4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.Linear(512, 1),
        # )

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, 64, kernel_size = 4, stride = 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            # torch.cat(condition,1)
            nn.Conv2d(64, 128, kernel_size = 4, stride = 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(128, 256, kernel_size = 4, stride = 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(256, 512, kernel_size = 4, stride = 2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(512, 512, kernel_size = 4, stride = 2),
            # nn.Sigmoid()
        )
        # pass


    def call(self, inputs):
        """
        Executes the discriminator model on a batch of input images and outputs whether it is real or fake.

        :param inputs: a batch of images, shape=[batch_size, height, width, channels]

        :return: a batch of values indicating whether the image is real or fake, shape=[batch_size, 1]
        """
        value = self.model(inputs)
        return value

    def loss_function(self, disc_real_output, disc_fake1_output, disc_fake2_output):
        """
        Outputs the discriminator loss given the discriminator model output on the real and generated images.

        :param disc_real_output: discriminator output on the real images, shape=[batch_size, 1]
        :param disc_fake_output: discriminator output on the generated images, shape=[batch_size, 1]

        :return: loss, the combined cross entropy loss, scalar
        """
      # TODO: Calculate the loss

        d_loss_real = np.mean(np.square(disc_real_output - 1.))
        d_loss_fake1 = np.mean(np.square(disc_fake1_output))
        d_loss_fake2 = np.mean(np.square(disc_fake2_output))

        loss = (1. / 2 * (d_loss_real + 1. / 2 * (d_loss_fake1 + d_loss_fake2)))

        # self.d_loss = (1. / 2 * (d_loss_real + 1. / 2 * (d_loss_fake1 + d_loss_fake2))) * self.gan_loss_weight
        # loss = 1/2 * nn.CrossEntropyLoss(np.square(disc_real_output - 1)) + 1/2 * nn.CrossEntropyLoss(np.square(disc_fake_output))
        return loss
