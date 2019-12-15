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
from datetime import datetime

# Killing optional CPU driver warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_available = torch.cuda.is_available()
print("GPU Available: ", gpu_available)


parser = argparse.ArgumentParser(description='IPCGAN')

parser.add_argument('--img-dir', type=str, default='./data/celebA',
                    help='Data where training images live')

parser.add_argument('--out-dir', type=str, default='/results',
                    help='Data where sampled output images will be written')

parser.add_argument('--saved_model_folder', type=str,
                    help='the path of folder which stores the parameters file',
                    default='/checkpoints')

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
def train(generator, discriminator, device, epoch):
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
    total_fid = 0
    train_size = int(args.n_images*0.2)
    for i in range (int(train_size/args.batch_size)):
        # print(torch.cuda.memory_cached(device)) 
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
        g_loss = generator.loss_function(g_output, batch, d_fake1_true, labels[:,0]) 
        g_loss.backward()
        generator.optimizer.step() 
        if i % 500 == 0:
            #make the axes match the original shape
            batch_fid =  np.moveaxis(np.asarray(batch.cpu().detach()), 1, 3) #swap axes
            gen_fid =  np.moveaxis(np.asarray(g_output.cpu().detach()), 1, 3) #swap axes
            current_fid = calculate_fid(batch_fid, gen_fid, use_multiprocessing = False, batch_size = args.batch_size)
            total_fid += current_fid 
            print('**** INCEPTION DISTANCE: %g ****' % current_fid) 
            # save_model(generator, discriminator, dir=args.saved_model_folder, filename='epoch_%d_iter_%d.pth'%(epoch, i))
            # print('checkpoint has been created!')
        if i % 25 == 0: 
            imgs =  np.moveaxis(np.asarray(g_output.cpu().detach()), 1, 3)[0:5]
            for k in range (5):  
                outdir =  os.getcwd() + args.out_dir
                if not os.path.exists(outdir):
                        os.mkdir(outdir)
                img = imgs[k] 
                img = (img+1)*127.5
                img = img.astype(np.uint8)
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                imwrite(outdir + '/res_%d.jpg' %((epoch*i)+k), img.astype(np.uint8) ) 
    avg_fid = total_fid/i
    return avg_fid, g_losses, d_losses


# Test the model by generating some samples.
def test(generator, device, source_img = "15_Daniel_Radcliffe_0003.jpg", target_label = 4):
    """
    Test the model by loading the newest checkpoint and generate sample images

    :param generator: generator model, device (cuda or cpu), source image and target age

    :return: None
    """ 
    #checkpoints = "checkpoints/IPCGANS/2019-01-14_08-34-45/gepoch_6_iter_500.pth" #find your favorite checkpoint and load it 
    print('========================== Testing ==========================')
    test_imgs = ["17_Daniel_Radcliffe_0007.jpg", "16_Emma_Watson_0009.jpg", "18_Chris_Brown_0014.jpg", "19_Robert_Pattinson_0008.jpg", "20_Dev_Patel_0004.jpg", "20_Josh_Peck_0006.jpg", "61_Robin_Williams_0013.jpg", "62_Mark_Hamill_0012.jpg", "61_Didi_Conn_0001.jpg"]
    target_ages = [4,4,4,4,4,4, 0, 0, 0]

    checkpoint_dir = "checkpoints/" #find your favorite checkpoint and load it
    paths = np.asarray(sorted(os.listdir(checkpoint_dir)))
    date_dir = checkpoint_dir+ paths[-1]  
    paths = np.asarray(sorted(os.listdir(date_dir)))
    generator_state = date_dir + '/' + paths[-1] #get the newest state in the last directory 

    #load the dictionary
    state_dict = torch.load(generator_state)
    generator = load_generator_state_dict(generator, state_dict)

    for i in range (len(test_imgs)):
        source_imgg = test_imgs[i]
        target_label = target_ages[i]
        source_img, target_labels = prep_test_im(source_imgg, target_label) 
        #convert to tensors
        target_labels = torch.tensor(target_labels, device =device).float()
        source_img = torch.tensor(source_img, device =device).float()

        generator.eval()
        with torch.no_grad():
            generate_image= generator(source_img, target_labels)

        image_dir  = 'data/CACD2000'  
        realimg = cv2.imread(os.path.join(image_dir, source_imgg))
        realimg = cv2.cvtColor(realimg, cv2.COLOR_BGR2RGB) 

        #prep output image    
        img =  np.moveaxis(np.asarray(generate_image.cpu().detach()), 1, 3)[0]
        img = (img+1)*127.5
        img = img.astype(np.uint8)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  
        outdir =  os.getcwd() + "/test_results"
        if not os.path.exists(outdir):
            os.mkdir(outdir) 
        imwrite(outdir + '/test_result_%d.jpg' %i, img.astype(np.uint8))  
        imwrite(outdir + '/test_original_%d.jpg' %i, realimg.astype(np.uint8))  
    return None

def prep_test_im(img_path, target_label):
    image_dir  = 'data/CACD2000'  
    img = cv2.imread(os.path.join(image_dir, img_path))
    if len(np.asarray(img).shape) > 0 :  
        img = cv2.resize(img, (args.image_size, args.image_size)) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = (img/127.5)-1
    imgs = [img, img]
    imgs = np.swapaxes(imgs, 1, 3)
    n = len(imgs)
    real_labels_onehot = np.zeros((n, 5, args.image_size, args.image_size))
    real_labels_onehot[np.arange(n), target_label, :,:] = np.ones((args.image_size,args.image_size)) 
    return imgs, real_labels_onehot

def save_model(generator, discriminator,dir,filename):
    outdir =  os.getcwd() + dir
    if not os.path.exists(outdir):
            os.mkdir(outdir)
    TIMESTAMP = "/{0:%Y-%m-%d_%H-%M-%S}".format(datetime.now())
    outdir = outdir + TIMESTAMP 
    if not os.path.exists(outdir):
            os.mkdir(outdir)
    torch.save(generator.state_dict(),os.path.join(outdir,"g"+filename))
    torch.save(discriminator.state_dict(),os.path.join(outdir,"d"+filename)) 

def load_generator_state_dict(generator,state_dict):
    pretrained_dict = state_dict
    # step2: get model state_dict
    model_dict = generator.state_dict()
    # step3: remove pretrained_dict params which is not in model_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # step4: update model_dict using pretrained_dict
    model_dict.update(pretrained_dict)
    # step5: update model using model_dict
    generator.load_state_dict(model_dict) 
    return generator  


## --------------------------------------------------------------------------------------

def main(): 
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
            avg_fid, g_losses, d_losses = train(generator, discriminator, device, epoch)
            print('========================== Average FID: %d  ==========================' % avg_fid)  
            save_model(generator, discriminator, dir=args.saved_model_folder, filename='epoch_%d.pth'%(epoch))
            print('checkpoint has been created!')
    if args.mode == 'test':
        test(generator, device) 
    test(generator, device) 
        
    # except RuntimeError as e:
    #     print(e)

if __name__ == '__main__':
   main()
