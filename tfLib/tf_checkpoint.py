import os
import numpy as np

import tensorflow as tf

# Developed for Tensorflow 2.4.1

class TF2Checkpoint():

    def __init__(self, checkpoint_dir:str, checkpoint:tf.train.Checkpoint):
        # Checkpoint - input (checkpoint_dir, checkpoint object)

        #checkpoint_dir = './eye_encoder_checkpoints'
        #checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        
        self.manager = tf.train.CheckpointManager(
            checkpoint, directory=checkpoint_dir, max_to_keep=5)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        #print(manager.checkpoints)
        self.val_losses_path = os.path.join(checkpoint_dir, "validation_losses.npy")
        if os.path.isfile(self.val_losses_path):
            self.val_losses = list(np.load(self.val_losses_path))
        else:
            self.val_losses = list()

    def update(self, valid_loss: float) -> None:
        """Updates the checkpoint and saves new checkpoint if validation loss has improved (the lower the better).
        """
        try:
            best_valid_loss = min(self.val_losses)
            if best_valid_loss > valid_loss:
                print(f"Validation score improved from {best_valid_loss} to {valid_loss}.")
                self.manager.save()
                
                self.val_losses.append(valid_loss)
                np.save(self.val_losses_path, np.array(self.val_losses))
            else:
                print(f"Validation score did not improve, best validation score: {best_valid_loss}, current validation score: {valid_loss}")
        except:
            self.val_losses.append(valid_loss)

    def save_model():
        pass

if __name__ == "__main__":
    # Use case
    checkpoint_dir = './eye_encoder_checkpoints'
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        eye_encoder=eye_encoder,
        eye_decoder=eye_decoder
    )

    tf2_checkpoint = TF2Checkpoint(checkpoint_dir, checkpoint)

    for i in range(1):
        # train
        tf2_checkpoint.update(validation_loss)