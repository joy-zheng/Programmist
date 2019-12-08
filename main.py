import tensorflow as tf
from tensorflow.keras import Model
from preprocess import get_metadata, get_image
from generator import Generator_Model
from discriminator import Distcriminator_Model
import tensorflow_gan as tfgan
import tensorflow_hub as hub
import numpy as np

batch_size = 30
z_dim = 500



# Train the model for one epoch.
def train(generator, discriminator, dataset_iterator, manager):
    """
    Train the model for one epoch. Save a checkpoint every 500 or so batches.

    :param generator: generator model
    :param discriminator: discriminator model
    :param dataset_ierator: iterator over dataset, see preprocess.py for more information
    :param manager: the manager that handles saving checkpoints by calling save()

    :return: The average FID score over the epoch
    """
    # Loop over our data until we run out
    for iteration, batch in enumerate(dataset_iterator):
        # TODO: Train the model
        noise = tf.random.uniform([batch_size, z_dim], -1, 1)
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            g_output = generator(noise)
            d_fake = discriminator(g_output)
            d_real = discriminator(batch)
            g_loss = generator.loss_function(d_fake)
            d_loss = discriminator.loss_function(d_fake, d_real)
        g_gradients = g_tape.gradient(g_loss,  generator.trainable_variables)
        generator.optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))        
        d_gradients = d_tape.gradient(d_loss,  discriminator.trainable_variables)
        discriminator.optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))


# Test the model by generating some samples.
def test(generator):
    """
    Test the model.

    :param generator: generator model

    :return: None
    """
    # TODO: Replace 'None' with code to sample a batch of random images
    img = tf.random.uniform([args.batch_size, args.z_dim], -1, 1)
    img = generator(img)
    

    ### Below, we've already provided code to save these generated images to files on disk
    # Rescale the image from (-1, 1) to (0, 255)
    img = ((img / 2) - 0.5) * 255
    # Convert to uint8
    img = img.astype(np.uint8)
    # Save images to disk
    for i in range(0, args.batch_size):
        img_i = img[i]
        s = args.out_dir+'/'+str(i)+'.png'
        imwrite(s, img_i)

## --------------------------------------------------------------------------------------

def main():
    # Load a batch of images (to feed to the discriminator)

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