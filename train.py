from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.compat.v1.image import crop_and_resize

from tfLib.advloss import get_adversarial_loss
from tfLib.loss import L1, SSCE
from Dataset import Dataset2
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

def compute_loss(x, xm, x_left_p, x_right_p, generator, discriminator, encoder):
    """Construct training pipeline from x -> y
    """

    # xm - mask where pixel value = 1 where eye region is located
    # xc - full image (pixels in eye region = 0)
    xc = x * (1 - xm)  #corrputed images

    xl_left, xl_right = crop_resize(x, x_left_p, x_right_p)

    # Extract eye features
    xl_left_fp  = encoder(xl_left)
    xl_right_fp = encoder(xl_right)

    # Generate gaze-corrected image
    yo = generator([xc, xm, xl_left_fp, xl_right_fp])

    # Extract eyes of generated image?
    yl_left, yl_right = crop_resize(yo, x_left_p, x_right_p)

    # Extract eyes into features?
    yl_left_fp = encoder(yl_left)
    yl_right_fp = encoder(yl_right)

    y = xc + yo * xm

    if opt.is_ss:

        d_logits, d_logits_left, d_logits_right = discriminator([x, xl_left, xl_right, xl_left_fp, xl_right_fp]) # Real images -> Discriminator loss
        g_logits, g_logits_left, g_logits_right = discriminator([y, yl_left, yl_right, yl_left_fp, yl_right_fp]) # Fake images -> Generator loss

        # Sparse Softmax Cross entropy - self supervision of eye eye embeedding?
        r_cls_loss = SSCE(labels=tf.zeros(shape=[opt.batch_size], dtype=tf.int32), logits=d_logits_left) + \
                    SSCE(labels=tf.ones(shape=[opt.batch_size], dtype=tf.int32), logits=d_logits_right)
        f_cls_loss = SSCE(labels=tf.zeros(shape=[opt.batch_size], dtype=tf.int32), logits=g_logits_left) + \
                    SSCE(labels=tf.ones(shape=[opt.batch_size], dtype=tf.int32), logits=g_logits_right)

    else:
        d_logits = discriminator([x, xl_left, xl_right, xl_left_fp, xl_right_fp])
        g_logits = discriminator([y, yl_left, yl_right, yl_left_fp, yl_right_fp])

    d_loss_fun, g_loss_fun = get_adversarial_loss(opt.loss_type)
    d_gan_loss = d_loss_fun(d_logits, g_logits)
    g_gan_loss = g_loss_fun(g_logits)

    # L1 for eye region reconstruction
    percep_loss = L1(xl_left_fp, yl_left_fp) + L1(xl_right_fp, yl_right_fp)
    
    # L1 for face reconstruction
    recon_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(y - x),
                        axis=[1, 2, 3]) / (opt.crop_w * opt.crop_h * opt.output_nc))

    if opt.is_ss:
        D_loss = d_gan_loss \
            + opt.lam_ss * r_cls_loss
        G_loss = g_gan_loss + opt.lam_r * recon_loss + opt.lam_p * percep_loss \
            + opt.lam_ss * f_cls_loss
    else:
        D_loss = d_gan_loss
        G_loss = g_gan_loss + opt.lam_r * recon_loss + opt.lam_p * percep_loss

    return D_loss, G_loss

def train_step(train_dataset, generator, discriminator, eye_encoder):
    start_time = time.time()
    d_losses = list()
    g_losses = list()
    t_iter = tqdm(train_dataset)
    for step, (x, eye_pos) in enumerate(t_iter):
    
        xm, x_left_p, x_right_p = get_Mask_and_pos(eye_pos)
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape(persistent=True) as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            loss_d, loss_g = compute_loss(x, xm, x_left_p, x_right_p, generator, discriminator, eye_encoder)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads_d = tape.gradient(loss_d, discriminator.trainable_weights)
        grads_g = tape.gradient(loss_g, generator.trainable_weights)
        del tape

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer_d.apply_gradients(zip(grads_d, discriminator.trainable_weights))
        optimizer_g.apply_gradients(zip(grads_g, generator.trainable_weights + eye_encoder.trainable_weights))
        #optimizer_g.apply_gradients(zip(grads_g, generator.trainable_weights))

        d_losses.append(float(loss_d))
        g_losses.append(float(loss_g))
        t_iter.set_description_str(f"Train loss: D - {np.mean(d_losses)}, G - {np.mean(g_losses)}")

    return np.mean(d_losses), np.mean(g_losses), time.time() - start_time

def validation_step(test_dataset, generator, discriminator, eye_encoder):
    start_time = time.time()
    d_losses = list()
    g_losses = list()
    t_iter = tqdm(test_dataset)
    for step, (x, eye_pos) in enumerate(t_iter):
        xm, x_left_p, x_right_p = get_Mask_and_pos(eye_pos)
        
        loss_d, loss_g = compute_loss(x, xm, x_left_p, x_right_p, generator, discriminator, eye_encoder)

        d_losses.append(float(loss_d))
        g_losses.append(float(loss_g))
        t_iter.set_description_str(f"Test loss: D - {np.mean(d_losses)}, G - {np.mean(g_losses)}")
    return np.mean(d_losses), np.mean(g_losses), time.time() - start_time

def visualize(x, y, y_real, save_path, num_samples=10):
    """Given input image x (B,H,W,3) and output image y (B,H,W,3).
    Visualize x, y side by side.
    
    """

    fig, axs = plt.subplots(num_samples,3, figsize=(10,num_samples*5))
    for i in range(num_samples):
        x_temp = ((x+1)/2)[i]
        y_temp = ((y+1)/2)[i]
        y_real_temp = ((y_real+1)/2)[i]

        axs[i][0].imshow(x_temp)
        axs[i][1].imshow(y_temp)
        axs[i][2].imshow(y_real_temp)
    #plt.show()
    fig.savefig(save_path)

def inference(x, xm, x_left_p, x_right_p, generator, encoder):
    """Construct training pipeline from x -> y
    """

    # xm - mask where pixel value = 1 where eye region is located
    # xc - full image (pixels in eye region = 0)
    xc = x * (1 - xm)  #corrputed images

    xl_left, xl_right = crop_resize(x, x_left_p, x_right_p)

    # Extract eye features
    xl_left_fp  = encoder(xl_left)
    xl_right_fp = encoder(xl_right)

    # Generate gaze-corrected image
    yo = generator([xc, xm, xl_left_fp, xl_right_fp]) 

    return yo, xc + yo*xm

def visualize_batch(test_dataset, generator, eye_encoder):
    for x, eye_pos in test_dataset.take(1):
        xm, x_left_p, x_right_p = get_Mask_and_pos(eye_pos)
        
        save_path = f"gaze_correcter_checkpoints/test_samples{str(epoch).rjust(2, '0')}"

        y, y_real = inference(x, xm, x_left_p, x_right_p, generator, eye_encoder)
        visualize(x, y, y_real, save_path, num_samples=7)

if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    eye_encoder, _, generator, discriminator = Gaze_GAN2(opt).build_train_model()
    #eye_encoder = tf.keras.models.load_model("eye_encoder/")
    eye_encoder = tf.keras.models.load_model("notebooks/models/eye_encoder", compile=False)
    dataset = Dataset2(opt)
    train_dataset, test_dataset = dataset.input()

    #print(eye_encoder.summary())
    #print(generator.summary())
    #print(discriminator.summary())

    optimizer_d = tf.keras.optimizers.Adam(opt.lr_d * 1, beta_1=opt.beta1, beta_2=opt.beta2) # lr_decay = 1 ?
    optimizer_g = tf.keras.optimizers.Adam(opt.lr_g * 1, beta_1=opt.beta1, beta_2=opt.beta2) # lr_decay = 1 ?

    checkpoint_dir = 'gaze_correcter_checkpoints'
    checkpoint = tf.train.Checkpoint(
        eye_encoder=eye_encoder,
        generator=generator,
        discriminator=discriminator,
        optimizer_d=optimizer_d,
        optimizer_g=optimizer_g
    )
    manager = tf.train.CheckpointManager(checkpoint, directory=checkpoint_dir, max_to_keep=5)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    epochs = 100
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        train_d_loss, train_g_loss, train_time = train_step(train_dataset, generator, discriminator, eye_encoder)
        valid_d_loss, valid_g_loss, valid_time = validation_step(test_dataset, generator, discriminator, eye_encoder)

        manager.save()
        
        # Saves inference visualization of input and output
        visualize_batch(test_dataset, generator, eye_encoder)