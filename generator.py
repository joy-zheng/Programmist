import sys
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import preprocess
import dgl

class Generator_Model(nn.Module):
    def __init__(self):
        """
        The model for the generator network is defined here.
        """
        super(Generator_Model, self).__init__()
        # TODO: Define the model, loss, and optimizer
        pass

    def call(self, inputs):
        """
        Executes the generator model on the random noise vectors.

        :param inputs: a batch of random noise vectors, shape=[batch_size, z_dim]

        :return: prescaled generated images, shape=[batch_size, height, width, channel]
        """
        # TODO: Call the forward pass
        pass

    def loss_function(self, disc_fake_output):
        """
        Outputs the loss given the discriminator output on the generated images.

        :param disc_fake_output: the discrimator output on the generated images, shape=[batch_size,1]

        :return: loss, the cross entropy loss, scalar
        """
        # TODO: Calculate the loss
        pass

    def identity_preserving_module():
        pass
    def age_classification_module():
        pass
