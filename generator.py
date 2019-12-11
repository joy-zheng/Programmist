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
        self.iteration = 0
        
        # Initialize hyperparameters
        self.learning_rate = 5e-4
        self.batch_size = 10
        self.epochs = 15
        self.generator_weight = 80 # TODO tweak it
        self.age_weight = 25 # TODO tweak it
        self.identity_weight = 5e-4 # TODO tweak it
    
        # Initialize layers
        self.conv1 = nn.Conv2d(8, 32, kernel_size=7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(32, eps=0.001)
        self.bn2 = nn.BatchNorm2d(64, eps=0.001)
        self.bn3 = nn.BatchNorm2d(128, eps=0.001)
        self.bn4 = nn.BatchNorm2d(64, eps=0.001)
        self.bn5 = nn.BatchNorm2d(32, eps=0.001)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=7, stride=1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        self.tanh = nn.Tanh()
        self.res_block1 = models.resnet.BasicBlock(128, 128)
        self.res_block2 = models.resnet.BasicBlock(128, 128)
        self.res_block3 = models.resnet.BasicBlock(128, 128)
        self.res_block4 = models.resnet.BasicBlock(128, 128)
        self.res_block5 = models.resnet.BasicBlock(128, 128)
        self.res_block6 = models.resnet.BasicBlock(128, 128)

    def forward(self, inputs, labels):
        """
        Executes the generator model on the random noise vectors.
        :param inputs: images and conditional feature maps concatenated together.
        :return: prescaled generated images, shape=[batch_size, height, width, channel]
        """
        # TODO: Call the forward pass
        x = torch.cat((inputs, labels), 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # print("Conv 1 Shape:", x.shape)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # print("Conv 2 Shape:", x.shape)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        # print("Conv 3 Shape:", x.shape)

        # residual blocks x6 (as mentioned in section 3.3 of paper)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)

        x = self.deconv1(x)
        x = self.bn4(x)
        x = self.relu(x)
        # print("Deconv 1 Shape:", x.shape)
        x = self.deconv2(x)
        x = self.bn5(x)
        x = self.relu(x)
        # print("Deconv 2 Shape:", x.shape)
        
        x = self.conv4(x)
        x = self.tanh(x)
        return x

    def loss_function(self, real_img, fake_img, fake_age):
        """
        Outputs the loss given the discriminator output on the generated images.
        :param disc_fake_output: the discrimator output on the generated images, 
        :return: loss, the cross entropy loss, scalar
        """
        # TODO: Calculate the loss
        # fake_img = self.call(real_img, target_ae_group)
        generator_loss = (1 / 2 * torch.mean((fake_img - 1).pow(2)))
        # print('age label shape', fake_age.shape)
        age_loss = self.calculate_age_loss(fake_img, fake_age) 
        identity_loss = self.identity_preserving_module(real_img, fake_img)
        print('**** At iteration ', self.iteration)
        print('*** age_loss: ', age_loss)
        print('*** identity_loss: ', identity_loss)
        print('*** pure_generator_loss: ', generator_loss)
        weighted_loss = self.generator_weight * generator_loss + self.age_weight * age_loss + self.identity_weight * identity_loss
        print("Generator loss:", weighted_loss)
        self.iteration += 1
        return weighted_loss
    
    def calculate_age_loss(self, fake_img, fake_age_labels):
        """
        Calculate age loss for generator
        :param batched_real_img: a batch of training images, shape=[batch_size, img_height, img_width, num_channel]
        :param target_age_grouup: a list of target age groups, shape=[batch_size,num_age_group]
        :return a flqoat loss for the batch
        """
        fake_age_logits = self.classify_age_alexnet(fake_img)
        age_loss = self.softmax_cross_entropy_loss(fake_age_logits, fake_age_labels)
        #    age_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        #                logits=fake_age, labels=target_age_group))
        return age_loss

    def classify_age_alexnet(self, fake_img):
        """
        Use a pre-trained alexnet age classifier to label the fake images with age groups
        :param fake_img: a set of fake images generated from the batch, shape=[batch_size, img_height, img_width, num_channel]
        :return fake_age, shape=[batch_size]
        """
        # https://pytorch.org/hub/pytorch_vision_alexnet/ for reference
        # print('in', fake_img.shape)
        # TODO fake_img = fake_img.permute()
        num_age_group = 5
        alexnet = models.alexnet(pretrained=True)
        alexnet.classifier[6] = nn.Linear(in_features=4096, out_features=num_age_group)
        alexnet.eval()
        prepare = transforms.Compose([

            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_stack = []
        
        for img in fake_img:
            sample = prepare(img)
            img_stack.append(sample)
        prepared_img = torch.stack(img_stack)
        output = alexnet(prepared_img)
        return output

    def softmax_cross_entropy_loss(self, logits, labels):
        """
        Calculate softmax cross entropy loss from given logits and labels
        
        :param logits: fake_age, shape=[batch_size]
        :param labels: target_age_grouup, shape=[batch_size, num_age_group]
        :return a float loss for the batch
        """
        # checkout https://discuss.pytorch.org/t/pytorch-equivalence-to-sparse-softmax-cross-entropy-with-logits-in-tensorflow/18727/2
        loss = nn.CrossEntropyLoss()
        # labels = labels.astype(np.float32)
        labels = torch.Tensor(labels).long()
        # print('lables',labels.shape)

        output = loss(logits, labels)
        return torch.mean(output)


    def identity_preserving_module(self, org_image, generated_image):
        """
        Outputs the identity loss for the given image and its generated image.
        :param disc_fake_output: original_features and generated_features from the 8th layer of AlexNet.
        :return: the sum of mean squared identity loss
        """
        original_features = self.alex_features(org_image)
        generated_features = self.alex_features(generated_image)
        # identity_loss =  0
        # for i in range(len(original_features)):
        #     identity_loss += torch.mean((abs(original_features[i] - generated_features[i])).pow(2)   )
        identity_loss = torch.nn.functional.mse_loss(original_features, generated_features)
        return identity_loss

    def alex_features (self, input_image):
        # print('*** identity alexnet input image: ', input_image.shape)
        alexnet_model = models.alexnet(pretrained=True)
        for param in alexnet_model.parameters():
            param.requires_grad = False
        alexnet_model.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )
        # set_parameter_requires_grad(alexnet_model, True)
        # num_ftrs = alexnet_model.classifier[6].in_features
        # alexnet_model.classifier[6] = nn.Linear(num_ftrs,5)
        process = transforms.Compose([
            # transforms.Resize(256),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # input_tensor = process(input_image)
        # input_batch = input_tensor.unsqueeze(0)
        # with torch.no_grad():
        #     output = alexnet_model(input_batch)
        img_stack = []
        for img in input_image:
            sample = process(img)
            img_stack.append(sample)
        prepared_img = torch.stack(img_stack)
        output = alexnet_model.features(prepared_img)
        # print('*** identity alexnet output features:', output.shape)
        return output