from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import datetime
from pathlib import Path

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.compat.v1.image import crop_and_resize

from tfLib.advloss import get_adversarial_loss
from tfLib.loss import L1
from Dataset import Dataset2, Dataset
from GazeGAN import Gaze_GAN2
from config.train_options import TrainOptions
from tfLib.tf_checkpoint import TF2Checkpoint
opt = TrainOptions().parse()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)

# Fixes "InternalError: Blas SGEMM launch failed"
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print("Invalid device or cannot modify virtual devices once initialized.")  

# Data preprocessing for mask
def get_Mask_and_pos(eye_pos, flag=0):
    """ Given eye_pos (shape=[4,]), 
        returns masks for cropping eyes. 
    """
    batch_mask = []
    batch_left_eye_pos = []
    batch_right_eye_pos = []
    for i in range(opt.batch_size):

        current_eye_pos = eye_pos[i]
        left_eye_pos = []
        right_eye_pos = []

        if flag == 0:

            mask = np.zeros(shape=[opt.img_size, opt.img_size, opt.output_nc])
            scale = current_eye_pos[1] - 15
            down_scale = current_eye_pos[1] + 15
            l1_1 =int(scale)
            u1_1 =int(down_scale)
            #x
            scale = current_eye_pos[0] - 25
            down_scale = current_eye_pos[0] + 25
            l1_2 = int(scale)
            u1_2 = int(down_scale)

            mask[l1_1:u1_1, l1_2:u1_2, :] = 1.0
            left_eye_pos.append(float(l1_1)/opt.img_size)
            left_eye_pos.append(float(l1_2)/opt.img_size)
            left_eye_pos.append(float(u1_1)/opt.img_size)
            left_eye_pos.append(float(u1_2)/opt.img_size)

            scale = current_eye_pos[3] - 15
            down_scale = current_eye_pos[3] + 15
            l2_1 = int(scale)
            u2_1 = int(down_scale)

            scale = current_eye_pos[2] - 25
            down_scale = current_eye_pos[2] + 25
            l2_2 = int(scale)
            u2_2 = int(down_scale)

            mask[l2_1:u2_1, l2_2:u2_2, :] = 1.0

            right_eye_pos.append(float(l2_1) / opt.img_size)
            right_eye_pos.append(float(l2_2) / opt.img_size)
            right_eye_pos.append(float(u2_1) / opt.img_size)
            right_eye_pos.append(float(u2_2) / opt.img_size)

        batch_mask.append(mask)
        batch_left_eye_pos.append(left_eye_pos)
        batch_right_eye_pos.append(right_eye_pos)

    return np.array(batch_mask), np.array(batch_left_eye_pos), np.array(batch_right_eye_pos)

def crop_resize(input, boxes_left, boxes_right):

    shape = [int(item) for item in input.shape.as_list()]
    return crop_and_resize(input, boxes=boxes_left, box_ind=list(range(0, shape[0])),
                                    crop_size=[int(shape[-3] / 2), int(shape[-2] / 2)]), \
            crop_and_resize(input, boxes=boxes_right, box_ind=list(range(0, shape[0])),
                                crop_size=[int(shape[-3] / 2), int(shape[-2] / 2)])

def compute_loss(x, x_left_p, x_right_p, encoder, decoder):
    """Construct training pipeline from x -> y
    """
    xl_left, xl_right = crop_resize(x, x_left_p, x_right_p)
    xl_left_mirrored, xl_right_mirrored = tf.image.flip_left_right(xl_left), tf.image.flip_left_right(xl_right)

    # Extract eye features
    xl_left_fp  = encoder(xl_left)
    xl_right_fp = encoder(xl_right)
    xl_left_fp_mirrored  = encoder(xl_left_mirrored)
    xl_right_fp_mirrored = encoder(xl_right_mirrored)

    # Generate gaze-corrected image
    yl_left = decoder(xl_left_fp)
    yl_right = decoder(xl_right_fp)
    yl_left_mirrored = decoder(xl_left_fp_mirrored)
    yl_right_mirrored = decoder(xl_right_fp_mirrored)

    # L1 for eye region reconstruction
    percep_loss = L1(xl_left, yl_left)                   + L1(xl_right, yl_right) \
                + L1(xl_left, yl_left_mirrored)          + L1(xl_right, yl_right_mirrored) \
                + L1(xl_left_mirrored, yl_left)          + L1(xl_right_mirrored, yl_right) \
                + L1(xl_left_mirrored, yl_left_mirrored) + L1(xl_right_mirrored, yl_right_mirrored)

    return percep_loss

#@tf.function
def train_step(train_dataset, eye_encoder, eye_decoder):
    start = time.time()
    losses = list()
    t_iter = tqdm(train_dataset)
    for step, (x, eye_pos) in enumerate(t_iter):
        _, x_left_p, x_right_p = get_Mask_and_pos(eye_pos)
        
        with tf.GradientTape() as tape:
            
            # Compute the loss value for this minibatch.
            percep_loss = compute_loss(x, x_left_p, x_right_p, eye_encoder, eye_decoder)

        # Use the gradient tape to automatically retrieve gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(percep_loss, eye_encoder.trainable_weights + eye_decoder.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, eye_encoder.trainable_weights + eye_decoder.trainable_weights))

        losses.append(float(percep_loss))
        t_iter.set_description_str(f"Train loss (for one batch): {np.mean(losses)}")

        # Log every 200 batches.
        #if step % 200 == 0:
        #    print(
        #        "Training loss (for one batch) at step %d: %.4f"
        #        % (step, float(percep_loss))
        #    )
        #    print("Seen so far: %s samples" % ((step + 1) * opt.batch_size))

    return np.mean(losses), time.time() - start

#@tf.function
def validation_step(test_dataset, eye_encoder, eye_decoder):
    start = time.time()
    losses = list()
    t_iter = tqdm(test_dataset)
    for step, (x, eye_pos) in enumerate(t_iter):
        _, x_left_p, x_right_p = get_Mask_and_pos(eye_pos)
        percep_loss = compute_loss(x, x_left_p, x_right_p, eye_encoder, eye_decoder)

        losses.append(float(percep_loss))
        t_iter.set_description_str(f"Test loss (for one batch): {np.mean(losses)}")
        # Log every 200 batches.
        #if step % 200 == 0:
        #    print(
        #        "Test loss (for one batch) at step %d: %.4f"
        #        % (step, float(percep_loss))
        #    )
        #    print("Seen so far: %s samples" % ((step + 1) * opt.batch_size))

    return np.mean(losses), time.time() - start

def visualize(x, y, save_path, num_samples=3):
    """Given input image x (B,H,W,3) and output image y (B,H,W,3).
    Visualize x, y side by side.
    
    """
    fig, axs = plt.subplots(num_samples,2, figsize=(20,num_samples*10))
    for i in range(num_samples):
        x_temp = ((x+1)/2)[i]
        y_temp = ((y+1)/2)[i]

        axs[i][0].imshow(x_temp)
        axs[i][1].imshow(y_temp)
    #plt.show()
    fig.savefig(save_path)
    
def inference(x, eye_encoder, eye_decoder):
    return eye_decoder(eye_encoder(x))

def visualize_batch(x, eye_encoder, eye_decoder, save_path):
    
    y = inference(x, eye_encoder, eye_decoder)
    visualize(x, y, save_path)

if __name__ == "__main__":
    print(opt)

    opt.batch_size = 16

    eye_encoder, eye_decoder, _, _ = Gaze_GAN2(opt).build_train_model()

    dataset = Dataset(opt)
    train_dataset, test_dataset = dataset.input()

    print(eye_encoder.summary())
    print(eye_decoder.summary())

    optimizer = tf.keras.optimizers.Adam(opt.lr_d * 1, beta_1=opt.beta1, beta_2=opt.beta2) # lr_decay = 1 ?

    # Checkpoint - input (checkpoint_dir, checkpoint object)
    """One method for saving, and one constructor
    One for convert to SavedModel?
    """
    
    checkpoint_dir = './eye_encoder_checkpoints'
    Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)
    #checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = TF2Checkpoint(checkpoint_dir, tf.train.Checkpoint(
        optimizer=optimizer,
        eye_encoder=eye_encoder,
        eye_decoder=eye_decoder
        ))
    
    # Tensorboard
    log_dir="logs/"

    summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    epochs = 100
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        train_loss, train_time = train_step(train_dataset, eye_encoder, eye_decoder)
        valid_loss, valid_time = validation_step(test_dataset, eye_encoder, eye_decoder)

        checkpoint.update(valid_loss)

        for (x, eye_pos) in test_dataset.take(1):
            _, x_left_p, x_right_p = get_Mask_and_pos(eye_pos)
            xl_left, xl_right = crop_resize(x, x_left_p, x_right_p)

            save_path = f"eye_encoder_checkpoints/test_samples{str(epoch).rjust(2, '0')}"

            visualize_batch(xl_left, eye_encoder, eye_decoder, save_path)
            visualize_batch(xl_right, eye_encoder, eye_decoder, save_path)

        with summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss, step=epoch)
            tf.summary.scalar('valid_loss', valid_loss, step=epoch)
            tf.summary.scalar('train_time', train_time, step=epoch)
            tf.summary.scalar('valid_time', valid_time, step=epoch)
 


        

        