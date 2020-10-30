import os
import time

import numpy as np
import tensorflow as tf
from scipy import spatial


def pre_load_checkpoint(checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        # print(" [*] Reading checkpoint from {}".format(ckpt.model_checkpoint_path))
        epoch_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        return epoch_step, ckpt.model_checkpoint_path
    else:
        return 0, None