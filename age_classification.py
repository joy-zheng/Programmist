import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
from torchVision import transforms, models


def calculate_age_loss(batched_real_img, target_age_group):
    """
    Calculate age loss for generator

    :param batched_real_img: a batch of training images, shape=[batch_size, img_height, img_width, num_channel]

    :param target_age_grouup: a list of target age groups, shape=[num_age_group]

    :return a float loss for the batch

    """
    fake_img = generate_img(batched_real_img, target_age_group)
    fake_age = classify_age_alexnet(fake_img)
    age_loss = softmax_cross_entropy_loss(fake_age, target_age_group)
    #    age_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #                logits=fake_age, labels=target_age_group))
    return age_loss
 
def softmax_cross_entropy_loss(logits, labels):
    """
    Calculate softmax cross entropy loss from given logits and labels
    
    :param logits: fake_age, shape=[batch_size, num_age_group]

    :param labels: target_age_grouup, shape=[num_age_group]

    :return a float loss for the batch
    """
    # checkout https://discuss.pytorch.org/t/pytorch-equivalence-to-sparse-softmax-cross-entropy-with-logits-in-tensorflow/18727/2
    loss = nn.CrossEtnropyLoss()
    output = loss(logits, labels)
    return torch.mean(output)

 
def generate_img(batched_real_img, target_age_group):
    """
    Generate fake images that fall into target_age_group

    :param batched_real_img: a batch of training images, shape=[batch_size, img_height, img_width, num_channel]

    :param target_age_grouup: a list of target age groups, shape=[num_age_group]

    :return fake images, shape=[batch_size, img_height, img_width, num_channel]
    """
    # TODO: Call the forward pass from generator class
    pass
 
def classify_age_alexnet(fake_img):
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
    processed_img = torch.tensor(processed_img)
    output = alexnet(processed_img)
    _, pred = torch.max(output,1)
    return pred
    
