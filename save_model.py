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

if __name__ == "__main__":

    eye_encoder, _, generator, discriminator = Gaze_GAN2(opt).build_train_model()
    eye_encoder = tf.keras.models.load_model("notebooks/models/eye_encoder", compile=False)

    print(eye_encoder.summary())
    print(generator.summary())
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

    save_path = "models/gaze_corrector"
    generator._name = 'gaze_corrector'
    generator.save(save_path)

    save_path = "models/eye_encoder"
    eye_encoder._name = 'eye_encoder'
    eye_encoder.save(save_path)