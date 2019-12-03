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

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

        # self.main = nn.Sequential(
        #     # input is (nc) x 64 x 64
        #     nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf) x 32 x 32
        #     nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 2),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*2) x 16 x 16
        #     nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 4),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*4) x 8 x 8
        #     nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(ndf * 8),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     # state size. (ndf*8) x 4 x 4
        #     nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
        #     nn.Sigmoid()
        # )
        # pass


    def call(self, inputs):
        """
        Executes the discriminator model on a batch of input images and outputs whether it is real or fake.

        :param inputs: a batch of images, shape=[batch_size, height, width, channels]

        :return: a batch of values indicating whether the image is real or fake, shape=[batch_size, 1]
        """
        value = self.model(inputs)
        return value

    def loss_function(self, disc_real_output, disc_fake_output):
        """
        Outputs the discriminator loss given the discriminator model output on the real and generated images.

        :param disc_real_output: discriminator output on the real images, shape=[batch_size, 1]
        :param disc_fake_output: discriminator output on the generated images, shape=[batch_size, 1]

        :return: loss, the combined cross entropy loss, scalar
        """
      # TODO: Calculate the loss
        # # real image, right label
        # D1_logits = discriminator(imgs, name='discriminator', condition=true_label_fea_64)
        # # real image, false label
        # D2_logits = discriminator(imgs, name='discriminator', condition=false_label_fea_64, reuse=True)
        # # fake image, true label
        # D3_logits = discriminator(self.g_source, name='discriminator', condition=true_label_fea_64, reuse=True)

        # d_loss_real = tf.reduce_mean(tf.square(D1_logits - 1.))
        # d_loss_fake1 = tf.reduce_mean(tf.square(D2_logits))
        # d_loss_fake2 = tf.reduce_mean(tf.square(D3_logits))

        # self.d_loss = (1. / 2 * (d_loss_real + 1. / 2 * (d_loss_fake1 + d_loss_fake2))) * self.gan_loss_weight
  
        loss = 1/2 * nn.CrossEntropyLoss(np.square(disc_real_output - 1)) + 1/2 * nn.CrossEntropyLoss(np.square(disc_fake_output))
        return loss
