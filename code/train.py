import tensorflow as tf
import os
import sys
import numpy as np
import argparse
import importlib
import socket
import h5py
import random
from tqdm import tqdm
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import model_utils
import data_provider
import transformations as trafo


parser = argparse.ArgumentParser()
parser.add_argument('--num_gpus', type=int, default=1, help='How many gpus to use')
parser.add_argument('--model', default='model_release')
parser.add_argument('--batch_size', type=int, default=50) # 1: 50
parser.add_argument('--num_point', type=int, default=2048)
parser.add_argument('--loss', default='cd')
parser.add_argument('--log_dir', default='log_dddmp')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 251]')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()
cluster_num = 40

###### load training data #######
f = h5py.File('../data/MN40_4096_train_data.h5')  # path of the training data
data = f['data'][:]
data = data[:, :, 0:3]  # if use normal and curvature, comment this line
# data = data[:, :, 0:6]  # use point normal as input to extract covariance

pts_num = data.shape[1]
data_dim = data.shape[-1]

file_size = data.shape[0]
print('File size is %d' % file_size)
####################################


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
NUM_GPUS = FLAGS.num_gpus
assert (BATCH_SIZE % NUM_GPUS == 0)
DEVICE_BATCH_SIZE = BATCH_SIZE / NUM_GPUS
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
LOSS = FLAGS.loss

MODEL = importlib.import_module(FLAGS.model)
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR))
os.system('cp train.py %s' % (LOG_DIR))

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    From tensorflow tutorial: cifar10/cifar10_multi_gpu_train.py
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        # for g, _ in grad_and_vars:
        for g, v in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train(palette):

    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            ###################################################
            ### placeholders  # triplet input
            data_train_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, data_dim))
            data_positive_train_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, data_dim))
            data_negative_train_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, data_dim))

            is_training_placeholder = tf.placeholder(tf.bool, shape=())

            cluster_mean_placeholder = tf.placeholder(tf.float32, shape=(cluster_num, 128))
            indices_placeholder = tf.placeholder(tf.int64, shape=(BATCH_SIZE))

            batch = tf.get_variable('batch', [], initializer=tf.constant_initializer(0),
                                    trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            ###################################################
            ### get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            ###################################################
            tower_grads = []
            feature_anchor_gpu = []
            net_anchor_gpu = []
            net_positive_gpu = []
            net_negative_gpu = []
            cluster_loss_gpu = []
            contrastive_loss_gpu = []
            total_loss_gpu = []
            l2_loss_gpu = []
            all_loss_gpu = []
            ### get model and loss
            for i in range(NUM_GPUS):
                with tf.variable_scope(tf.get_variable_scope(), reuse=bool(i!=0)):
                    with tf.device('/gpu:%d' % (i)), tf.name_scope('gpu_%d' % (i)) as scope:
                        pc_patch = tf.slice(data_train_placeholder,
                                            [i * DEVICE_BATCH_SIZE, 0, 0], [DEVICE_BATCH_SIZE, -1, -1])
                        pc_patch_positive = tf.slice(data_positive_train_placeholder,
                                             [i * DEVICE_BATCH_SIZE, 0, 0], [DEVICE_BATCH_SIZE, -1, -1])
                        pc_patch_negative = tf.slice(data_negative_train_placeholder,
                                             [i * DEVICE_BATCH_SIZE, 0, 0], [DEVICE_BATCH_SIZE, -1, -1])

                        pc_features, pc_net, _ = MODEL.get_model(pc_patch, is_training_placeholder, bn_decay)
                        tf.get_variable_scope().reuse_variables()
                        pc_features_positive, pc_net_positive, _ = MODEL.get_model(pc_patch_positive,
                                                                        is_training_placeholder, bn_decay)
                        pc_features_negative, pc_net_negative, _ = MODEL.get_model(pc_patch_negative,
                                                                                is_training_placeholder, bn_decay)

                        pc_indices = tf.slice(indices_placeholder, [i*DEVICE_BATCH_SIZE], [DEVICE_BATCH_SIZE])
                        pc_cluster_loss, _, _, pc_contrastive_loss, pc_total_loss = MODEL.get_loss(pc_net, pc_net_positive,
                                                                                    pc_net_negative, cluster_mean_placeholder,
                                                                                    pc_indices)
                        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        update_ops = tf.group(*update_op)
                        with tf.control_dependencies([update_ops]):
                            pc_l2_loss = tf.losses.get_regularization_loss()
                            pc_l2_loss = 0.00001 * pc_l2_loss

                        pc_all_loss = pc_total_loss + pc_l2_loss
                        grads = optimizer.compute_gradients(pc_all_loss)
                        tower_grads.append(grads)

                        cluster_loss_gpu.append(pc_cluster_loss)
                        contrastive_loss_gpu.append(pc_contrastive_loss)
                        total_loss_gpu.append(pc_total_loss)
                        l2_loss_gpu.append(pc_l2_loss)
                        all_loss_gpu.append(pc_all_loss)

                        feature_anchor_gpu.append(pc_features)

                        net_anchor_gpu.append(pc_net)
                        net_positive_gpu.append(pc_net_positive)
                        net_negative_gpu.append(pc_net_negative)

            cluster_loss = tf.reduce_mean(cluster_loss_gpu)
            contrastive_loss = tf.reduce_mean(contrastive_loss_gpu)
            total_loss = tf.reduce_mean(total_loss_gpu)
            l2_loss = tf.reduce_mean(l2_loss_gpu)
            all_loss = tf.reduce_mean(all_loss_gpu)


            tf.summary.scalar('cluster_loss', cluster_loss)
            tf.summary.scalar('contrastive_loss', contrastive_loss)
            tf.summary.scalar('l2_loss', l2_loss)
            tf.summary.scalar('total loss', all_loss)

            feature_anchor = tf.concat(feature_anchor_gpu, 0)
            net_anchor = tf.concat(net_anchor_gpu, 0)
            net_positive = tf.concat(net_positive_gpu, 0)
            net_negative = tf.concat(net_negative_gpu, 0)

            grads = average_gradients(tower_grads)
            apply_gradient_op = optimizer.apply_gradients(grads, global_step=batch)

            variable_averages = tf.train.ExponentialMovingAverage(bn_decay, batch)
            variables_averages_op = variable_averages.apply(tf.trainable_variables())
            train_op = tf.group(apply_gradient_op, variables_averages_op)


        saver = tf.train.Saver(max_to_keep=11)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_placeholder: True})

        restore_epoch, checkpoint_path = model_utils.pre_load_checkpoint(LOG_DIR)
        global LOG_FOUT
        if restore_epoch == 0:
            LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
            LOG_FOUT.write(str(socket.gethostname()) + '\n')
            LOG_FOUT.write(str(FLAGS) + '\n')
        else:
            LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
            saver.restore(sess, checkpoint_path)

        ops = {'pointclouds_pl': data_train_placeholder,
               'pointclouds_pl_positive': data_positive_train_placeholder,
               'pointclouds_pl_negative': data_negative_train_placeholder,
               'is_training_pl': is_training_placeholder,
               'buffer': cluster_mean_placeholder,
               'indics': indices_placeholder,
               'feature_anchor': feature_anchor,
               'net_anchor': net_anchor,
               'net_positive': net_positive,
               'net_negative': net_negative,
               'cluster_loss': cluster_loss,
               'contrastive_loss': contrastive_loss,
               'total_loss': total_loss,
               'l2_loss': l2_loss,
               'all_loss': all_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               }

        feature_buffer = np.random.rand(file_size, 128)
        feature_buffer = trafo.unit_vector(feature_buffer, axis=1)  # normalize
        for epoch in tqdm(range(restore_epoch, MAX_EPOCH)):
        # for epoch in tqdm(range(MAX_EPOCH)):
            print('******* EPOCH %03d *******' % (epoch))
            sys.stdout.flush()

            ## clustering
            # cluster the memory buffer
            cluster_pred = SpectralClustering(n_clusters=cluster_num, gamma=1).fit_predict(feature_buffer)
            # cluster_pred = KMeans(n_clusters=cluster_num).fit_predict(feature_buffer)
            # cluster_pred = AgglomerativeClustering(n_clusters=cluster_num).fit_predict(feature_buffer)
            cluster_mean = np.zeros(shape=(cluster_num, 128))
            for cluster_idx in range(cluster_num):
                indices = np.where(cluster_pred == cluster_idx)[0]
                cluster_avg = np.mean(feature_buffer[indices, :], axis=0)
                cluster_mean[cluster_idx, :] = cluster_avg
            cluster_mean = trafo.unit_vector(cluster_mean, axis=1)


            train_one_epoch(sess, cluster_mean, cluster_pred, data, ops, train_writer)

            ## get the feature buffer after each epoch
            list = range(pts_num)
            seed = random.sample(list, NUM_POINT)
            data_2048 = data[:, seed, :]
            num_batches = file_size // BATCH_SIZE
            feature_buffer = np.zeros(shape=(file_size, 128))
            for batch_idx in range(num_batches+1):
                if batch_idx != num_batches:
                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = (batch_idx + 1) * BATCH_SIZE
                else:
                    start_idx = file_size - BATCH_SIZE
                    end_idx = file_size
                data_input = data_2048[start_idx:end_idx, :, :]
                feed_dict = {ops['pointclouds_pl']: data_input,
                             ops['is_training_pl']: False, }
                features = sess.run([ops['net_anchor']], feed_dict=feed_dict)
                feature_buffer[start_idx:end_idx, :] = features[0]

            if epoch % 50 == 0:
                # start_time = time.time()
                # tsne_visualization_color(feature_buffer, cluster_pred, palette, epoch)
                # end_time = time.time()
                # print 'tsne running time is %f' % (end_time-start_time)

                # start_time = time.time()
                # pca_visualization_color(feature_buffer, cluster_pred, palette, epoch)
                # end_time = time.time()
                # print 'pca running time is %f' % (end_time-start_time)

            # if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step=epoch)
                print("Model saved in file: %s" % save_path)


def tsne_visualization_color(feature_bank, colors, palette, epoch):
    # feature_bank: (file_size, 128)

    # Random state.
    RS = 20150101
    features_proj = TSNE(random_state=RS).fit_transform(feature_bank)
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(features_proj[:, 0], features_proj[:, 1], s=5, c=palette[colors.astype(np.int)])
    img_path = BASE_DIR + '/' + LOG_DIR + '/tsne-generated-color-' + str(epoch) + '.png'
    plt.savefig(img_path, dpi=120)
    # plt.show()

def pca_visualization_color(feature_bank, colors, palette, epoch):
    # feature_bank: (file_size, 128)

    # Random state.
    RS = 20150101
    pca = PCA(n_components=2)
    pca.fit(feature_bank)
    features_proj = pca.fit_transform(feature_bank)
    # features_proj = TSNE(random_state=RS).fit_transform(feature_bank)
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(features_proj[:, 0], features_proj[:, 1], s=5, c=palette[colors.astype(np.int)])
    img_path = BASE_DIR + '/' + LOG_DIR + '/pca-generated-color-' + str(epoch) + '.png'
    plt.savefig(img_path, dpi=120)
    # plt.show()

def train_one_epoch(sess, cluster_mean, cluster_pred, data, ops, train_writer):
    # cluster_mean: (10, 128)
    # cluster_pred: (file_size) cluster label for each object

    is_training = True

    num_batches = file_size // BATCH_SIZE

    idx = np.arange(file_size)
    np.random.shuffle(idx)
    data_shuffle = data[idx, ...]
    cluster_pred = cluster_pred[idx, ...]

    # extract 2048 points
    list = range(pts_num)
    seed = random.sample(list, NUM_POINT)
    current_data_anchor = data_shuffle[:, seed, :]
    ######

    # generate positive sample with different point distribution
    list = range(pts_num)
    seed = random.sample(list, NUM_POINT)
    current_data_positive = data_shuffle[:, seed, :]

    # generate negative sample from other clusters
    current_data_negative = np.zeros(shape=(file_size, NUM_POINT, data_dim))
    for i in range(file_size):
        indices = np.where(cluster_pred != cluster_pred[i])[0]
        negative_index = np.random.choice(indices, 1)
        current_data_negative[i, :, :] = current_data_anchor[negative_index, :, :]

    cluster_loss_sum = 0
    contrastive_loss_sum = 0
    total_loss_sum = 0
    l2_loss_sum = 0
    all_loss_sum = 0
    for batch_idx in range(num_batches):

        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE

        # Augment batched point clouds by rotation and jittering
        rotated_data, _ = data_provider.rotate_point_cloud_and_gt(current_data_anchor[start_idx:end_idx, :, :])
        scaled_data, _, scales = data_provider.random_scale_point_cloud_and_gt(rotated_data, batch_gt=None,
                                                                               scale_low=0.8, scale_high=1.2)
        shift_data, _ = data_provider.shift_point_cloud_and_gt(scaled_data, batch_gt=None, shift_range=0.2)
        jitter_data_anchor = data_provider.jitter_perturbation_point_cloud(shift_data, sigma=0.005, clip=0.015)
        jitter_data_anchor = data_provider.rotate_perturbation_point_cloud(jitter_data_anchor, angle_sigma=0.03,
                                                                      angle_clip=0.09)

        # Augment batched point clouds by rotation and jittering
        rotated_data_positive, _ = data_provider.rotate_point_cloud_and_gt(current_data_positive[start_idx:end_idx, :, :])
        scaled_data_positive, _, scales = data_provider.random_scale_point_cloud_and_gt(rotated_data_positive, batch_gt=None,
                                                                               scale_low=0.8, scale_high=1.2)
        shift_data_positive, _ = data_provider.shift_point_cloud_and_gt(scaled_data_positive, batch_gt=None, shift_range=0.2)
        jitter_data_positive = data_provider.jitter_perturbation_point_cloud(shift_data_positive, sigma=0.005, clip=0.015)
        jitter_data_positive = data_provider.rotate_perturbation_point_cloud(jitter_data_positive, angle_sigma=0.03,
                                                                      angle_clip=0.09)

        # Augment batched point clouds by rotation and jittering
        rotated_data_negative, _ = data_provider.rotate_point_cloud_and_gt(
            current_data_negative[start_idx:end_idx, :, :])
        scaled_data_negative, _, scales = data_provider.random_scale_point_cloud_and_gt(rotated_data_negative,
                                                                                        batch_gt=None,
                                                                                        scale_low=0.8, scale_high=1.2)
        shift_data_negative, _ = data_provider.shift_point_cloud_and_gt(scaled_data_negative, batch_gt=None,
                                                                        shift_range=0.2)
        jitter_data_negative = data_provider.jitter_perturbation_point_cloud(shift_data_negative, sigma=0.005,
                                                                             clip=0.015)
        jitter_data_negative = data_provider.rotate_perturbation_point_cloud(jitter_data_negative, angle_sigma=0.03,
                                                                             angle_clip=0.09)

        indices = cluster_pred[start_idx:end_idx]
        feed_dict = {ops['pointclouds_pl']: jitter_data_anchor,
                     ops['pointclouds_pl_positive']: jitter_data_positive,
                     ops['pointclouds_pl_negative']: jitter_data_negative,
                     ops['is_training_pl']: is_training,
                     ops['buffer']: cluster_mean,
                     ops['indics']: indices
                     }
        summary, step, _, net_anchor, net_positive, net_negative, cluster_loss, contrastive_loss, total_loss, l2_loss, all_loss = sess.run(
                                                                    [ops['merged'], ops['step'], ops['train_op'],
                                                                    ops['net_anchor'], ops['net_positive'], ops['net_negative'],
                                                                     ops['cluster_loss'], ops['contrastive_loss'], ops['total_loss'],
                                                                    ops['l2_loss'], ops['all_loss']], feed_dict=feed_dict)

        train_writer.add_summary(summary, step)

        cluster_loss_sum += cluster_loss
        contrastive_loss_sum += contrastive_loss
        total_loss_sum += total_loss
        l2_loss_sum += l2_loss
        all_loss_sum += all_loss


    print('cluster_loss: %f' % (cluster_loss_sum / float(num_batches)))
    print('contrastive_loss: %f' % (contrastive_loss_sum / float(num_batches)))
    print('total_loss: %f' % (total_loss_sum / float(num_batches)))
    print('l2_loss: %f' % (l2_loss_sum / float(num_batches)))
    print('final_loss: %f' % (all_loss_sum / float(num_batches)))


if __name__ == "__main__":
    palette = np.array(sns.color_palette("hls", cluster_num))
    train(palette)