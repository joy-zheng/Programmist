import sys
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import preprocess
import dgl


# Train the model for one epoch.
def train(generator, discriminator, dataset_iterator):
    """
    Train the model for one epoch. Save a checkpoint every 500 or so batches.

    :param generator: generator model
    :param discriminator: discriminator model
    :param dataset_ierator: iterator over dataset, see preprocess.py for more information
    :param manager: the manager that handles saving checkpoints by calling save()

    :return:
    """

    pass

# Test the model by generating some samples.
def test(generator):
    """
    Test the model.

    :param generator: generator model

    :return: None
    """
    pass

## --------------------------------------------------------------------------------------

def main():
    # Initialize generator and discriminator models
    generator = Generator_Model()
    discriminator = Discriminator_Model()

    try:
        # Specify an invalid GPU device
        with tf.device('/device:' + args.device):
            if args.mode == 'train':
                for epoch in range(0, args.num_epochs):
                    print('========================== EPOCH %d  ==========================' % epoch)
                    avg_fid = train(generator, discriminator, dataset_iterator, manager)
                    print("Average FID for Epoch: " + str(avg_fid))
                    # Save at the end of the epoch, too
                    print("**** SAVING CHECKPOINT AT END OF EPOCH ****")
                    manager.save()
            if args.mode == 'test':
                test(generator)
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
   main()
