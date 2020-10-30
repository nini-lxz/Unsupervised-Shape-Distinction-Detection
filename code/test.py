import tensorflow as tf
import numpy as np
import argparse
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import model_utils


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='2', help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 1]')
parser.add_argument('--num_point', type=int, default=2048)
parser.add_argument('--log_dir', default='trained_model')

FLAGS = parser.parse_args()

GPU_INDEX = FLAGS.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_INDEX
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point

LOG_DIR = FLAGS.log_dir
MODEL = importlib.import_module(FLAGS.model)  # import network module

# TEST_FILE_PATH = 'test_2048.h5' # path of testing point clouds
#
# f = h5py.File(TEST_FILE_PATH)
# test_data = f['data'][:]
# name = f['name'][:]

test_data = np.loadtxt('/data/xzli/testing_data_examples/plant_0313_xyz.txt')  # [2048, 3]
test_data = np.expand_dims(test_data, axis=0)  # [1, 2048, 3]
name = []
name.append('plant_0313')

test_data_size = test_data.shape[0]
data_dim = test_data.shape[2]
print("test data size is %d" % test_data_size)


def evaluate():
    with tf.device('/gpu:' + str(GPU_INDEX)):
        test_data_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, data_dim))
        is_training_placeholder = tf.placeholder(tf.bool)

        per_point_val, _, spatial_weights = MODEL.get_model(test_data_placeholder, is_training_placeholder)

        saver = tf.train.Saver()

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    #_, restore_model_path = model_utils.pre_load_checkpoint(LOG_DIR)
    restore_model_path = LOG_DIR + '/model.ckpt-200'
    saver.restore(sess, restore_model_path)
    print("Model restored: %s" % restore_model_path)

    ops = {'pointclouds_pl': test_data_placeholder,
           'is_training_pl': is_training_placeholder,
           'per_point_val': per_point_val}
           # 'spatial_weights': spatial_weights}

    is_training = False

    num_batches = test_data_size // BATCH_SIZE
    test_data_size_inuse = int(num_batches * BATCH_SIZE)
    per_point_vals = np.zeros(shape=(test_data_size_inuse, NUM_POINT, 1))
    per_point_features = np.zeros(shape=(test_data_size_inuse, NUM_POINT, 128))

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        current_test_data = test_data[start_idx:end_idx, :, :]

        feed_dict = {ops['pointclouds_pl']: current_test_data,
                     ops['is_training_pl']: is_training,
                     }

        per_point_val = sess.run([ops['per_point_val']],
                                            feed_dict=feed_dict)

        per_point_features[start_idx:end_idx, :, :] = per_point_val[0]
        per_point_vals[start_idx:end_idx, :, 0] = np.max(per_point_val[0], axis=-1)

        # per_point_vals[start_idx:end_idx, :, 0] = np.mean(per_point_val[0], axis=-1)

        # per_point_vals[start_idx:end_idx, :, 0] = np.linalg.norm(per_point_val[0], ord=2, axis=-1)


    ## save detected point saliency probability
    output_path = LOG_DIR + '/' + 'saliency_prob/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in range(test_data_size):
        detected_saliency_prob = np.concatenate([test_data[i, :, :], per_point_vals[i, :, :]], axis=-1)
        model_name = name[i] + '.txt'
        model_path = os.path.join(output_path, model_name)
        np.savetxt(model_path, detected_saliency_prob, fmt='%.3f')


if __name__ == '__main__':
    with tf.Graph().as_default():
        evaluate()