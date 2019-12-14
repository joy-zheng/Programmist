from preprocess import Data_Processor
from generator import Generator_Model
from discriminator import Discriminator_Model
from prep_age_labels import save_age_paths
import numpy as np
from imageio import imwrite
import os
import argparse
import torch 
from eval.fid import *
from eval.inception import InceptionV3
import cv2

# batch_size = 30
# image_size = 128
# z_dim = 500
# n_images = 163446

# Killing optional CPU driver warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_available = torch.cuda.is_available()
print("GPU Available: ", gpu_available)


parser = argparse.ArgumentParser(description='IPCGAN')

parser.add_argument('--img-dir', type=str, default='./data/celebA',
                    help='Data where training images live')

parser.add_argument('--out-dir', type=str, default='/results',
                    help='Data where sampled output images will be written')

parser.add_argument('--mode', type=str, default='train',
                    help='Can be "train" or "test"')

parser.add_argument('--n-images', type=int, default=163446,
                    help='total input images')

parser.add_argument('--restore-checkpoint', action='store_true',
                    help='Use this flag if you want to resuming training from a previously-saved checkpoint')

parser.add_argument('--z-dim', type=int, default=100,
                    help='Dimensionality of the latent space')

parser.add_argument('--batch-size', type=int, default=32,
                    help='Sizes of image batches fed through the network')

parser.add_argument('--image-size', type=int, default=128,
                    help='dimension of the input images')

parser.add_argument('--num-data-threads', type=int, default=2,
                    help='Number of threads to use when loading & pre-processing training images')

parser.add_argument('--num-epochs', type=int, default=8,
                    help='Number of passes through the training data to make before stopping')

parser.add_argument('--learn-rate', type=float, default=0.0002,
                    help='Learning rate for Adam optimizer')

parser.add_argument('--beta1', type=float, default=0.5,
                    help='"beta1" parameter for Adam optimizer')

parser.add_argument('--num-gen-updates', type=int, default=2,
                    help='Number of generator updates per discriminator update')

parser.add_argument('--log-every', type=int, default=7,
                    help='Print losses after every [this many] training iterations')

parser.add_argument('--save-every', type=int, default=500,
                    help='Save the state of the network after every [this many] training iterations')

parser.add_argument('--device', type=str, default='GPU:0' if gpu_available else 'CPU:0',
                    help='specific the device of computation eg. CPU:0, GPU:0, GPU:1, GPU:2, ... ')

args = parser.parse_args()


# Train the model for one epoch.
def train(generator, discriminator, device):
    """
    Train the model for one epoch. Save a checkpoint every 500 or so batches.

    :param generator: generator model
    :param discriminator: discriminator model
    :param dataset_ierator: iterator over dataset, see preprocess.py for more information
    :param manager: the manager that handles saving checkpoints by calling save()

    :return: The average FID score over the epoch
    """
    # Loop over our data until we run out
    print(device)
    d_losses  = []
    g_losses  = [] 
    data_processor = Data_Processor(batch_size = args.batch_size, image_size = args.image_size, mode='train')
    target_agegroup = None
    total_fid = 0
    train_size = int(args.n_images*0.9)
    for i in range (int(train_size/args.batch_size)):
        # print(torch.cuda.memory_cached(device))
    # for iteration, batch in enumerate(dataset_iterator):
        batch, batch_real_labels, batch_fake_labels, labels = data_processor.get_next_batch_image()[0:4] #Fancy way of getting a new batch of imgs and labels
        # temp = np.moveaxis(batch[0], 0, 2)
        # temp = temp*255
        # temp = temp.astype(np.uint8)
        # print(temp)
        # cv2.imshow("", temp)
        # cv2.waitKey(0)

        batch = torch.tensor(batch, device =device).float()
        batch_real_labels = torch.tensor(batch_real_labels, device =device).float()
        batch_fake_labels = torch.tensor(batch_fake_labels, device =device).float()
        
        # training discriminator
        discriminator.optimizer.zero_grad() 
        g_output = generator(batch, batch_real_labels)

        #fake img, real label
        d_fake1_true = discriminator(g_output,  batch_real_labels)
        #real img, fake label
        d_fake2_false = discriminator(batch, batch_fake_labels)
        #real img, real label
        d_real_real = discriminator(batch, batch_real_labels)
 
        d_loss = discriminator.loss_function(d_real_real, d_fake1_true, d_fake2_false)
        d_loss.backward()
        discriminator.optimizer.step()
    
        generator.optimizer.zero_grad()
        g_output = generator(batch, batch_real_labels)
        #fake img, real label
        d_fake1_true = discriminator(g_output,  batch_real_labels)
        g_loss = generator.loss_function(batch, d_fake1_true, labels[:,0]) 
        g_loss.backward()
        generator.optimizer.step()
        # print(g_output.data.cpu().numpy())
        if i % 500 == 0:
            #make the axes match the original shape
            batch_fid =  np.moveaxis(np.asarray(batch.cpu().detach()), 1, 3) #swap axes
            gen_fid =  np.moveaxis(np.asarray(g_output.cpu().detach()), 1, 3) #swap axes
            current_fid = calculate_fid(batch_fid, gen_fid, use_multiprocessing = False, batch_size = args.batch_size)
            total_fid += current_fid 
            print('**** INCEPTION DISTANCE: %g ****' % current_fid) 
        if i % 4 == 0: 
            imgs =  np.moveaxis(np.asarray(g_output.cpu().detach()), 1, 3)[0:5]
            for k in range (5):  
                outdir =  os.getcwd() + args.out_dir
                if not os.path.exists(outdir):
                        os.mkdir(outdir)
                img = imgs[k] 
                img = (img+1)*127.5
                img = img.astype(np.uint8)
                imwrite(outdir + '/res_%d.jpg' %(i+k), img.astype(np.uint8) ) 
    avg_fid = total_fid/i
    return avg_fid, g_losses, d_losses


# Test the model by generating some samples.
def test(generator, discriminator):
    """
    Test the model.

    :param generator: generator model

    :return: None
    """
    # TODO: Replace 'None' with code to sample a batch of random images
    data_processor = Data_Processor(batch_size = args.batch_size, image_size = args.image_size, mode='test')
    # test_size = int(n_images*0.1)
    # for i in range (int(test_size/batch_size)):

    batch, batch_real_labels, batch_fake_labels, labels = data_processor.get_next_batch_image()[0:4] #Fancy way of getting a new batch of imgs and labels
    # comment out to  look at  inputs
    batch = torch.tensor(batch, device =device).float()
    batch_real_labels = torch.tensor(batch_real_labels, device =device).float()
    batch_fake_labels = torch.tensor(batch_fake_labels, device =device).float()

    img = generator(batch, batch_real_labels)


    ### Below, we've already provided code to save these generated images to files on disk
    # Rescale the image from (-1, 1) to (0, 255)
 
    imgs =  np.moveaxis(np.asarray(g_output.cpu().detach()), 1, 3)
    for k in range(imgs.shape[0]):  
        outdir =  os.getcwd() + args.out_dir
        if not os.path.exists(outdir):
                os.mkdir(outdir)
        img = imgs[k] 
        img = (img+1)*127.5
        img = img.astype(np.uint8)
        imwrite(outdir + '/res_%d.jpg' %(k), img.astype(np.uint8) ) 
    return None
## --------------------------------------------------------------------------------------

def main():
    # Load a batch of images (to feed to the discriminator)
    # Initialize generator and discriminator models
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = Generator_Model()
    discriminator = Discriminator_Model()
    save_age_paths() #creates a shuffled ages_paths.txt

    #Now send existing model to device.
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    #Now send input to device and so on.
    if args.mode == 'train':
        for epoch in range(args.num_epochs):
            print('========================== EPOCH %d  ==========================' % (epoch+1))
            avg_fid, g_losses, d_losses = train(generator, discriminator, device)
            print('========================== Average FID: %d  ==========================' % avg_fid)
        torch.save(generator.state_dict(), './')
    if args.mode == 'test':
        test(generator, discriminator)
    # try:
    #     # Specify an invalid GPU device
    #     with tf.device('/device:' + args.device):
    #         if args.mode == 'train':
    #             for epoch in range(0, args.num_epochs):
    #                 print('========================== EPOCH %d  ==========================' % epoch)
    #                 avg_fid = train(generator, discriminator, dataset_iterator, manager)
    #                 print("Average FID for Epoch: " + str(avg_fid))
    #                 # Save at the end of the epoch, too
    #                 print("**** SAVING CHECKPOINT AT END OF EPOCH ****")
    #                 manager.save()
        
    # except RuntimeError as e:
    #     print(e)

if __name__ == '__main__':
   main()
