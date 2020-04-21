import tensorflow as tf
import sys
import os
import math
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'sampling'))
sys.path.append(os.path.join(BASE_DIR, 'grouping'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import tf_sampling
from tf_grouping import query_ball_point, group_point
import tf_util
from xconv import xconv, xconv_cov, sa_conv, xconv_new, deconv_new, deconv, channel_spatial_attention
from pointnet_util import pointnet_sa_module, pointnet_fp_module
import rscnn_util
import pointfly as pf

def get_model(x, is_training=True, bn_decay=None):
    """ x: [batch_size, num _point, 3]
    return: [batch_size, feature_num]
    """
    batch_size = x.get_shape()[0].value
    num_point = x.get_shape()[1].value

    nsample = 16

    ########## get covariance matrix using query ball############
    indices, pts_cnt = query_ball_point(0.1, nsample, x, x)  # x: pts; x: qrs
    nn_pts = group_point(x, indices)
    cov_x = tf_util.tf_cov(nn_pts)  # [batch_size, num_point, 9]

    ########################## Conv #############################
    ### first layer
    P1 = int(num_point/2)
    C1 = 32
    C_pts_fts = C1 // 2
    depth_multiplier = 4
    indices = tf_sampling.farthest_point_sample(P1, x)
    x_layer1 = tf_sampling.gather_point(x, indices)  # (batch_size, num_point/2, 3)
    fts_xconv_1 = xconv_new(x, cov_x, x_layer1, 'xconv_1_', batch_size, K=nsample, radius=0.1, P=P1, C=C1,
                      C_pts_fts=C_pts_fts, is_training=is_training, with_X_transformation=True,
                      depth_multiplier=depth_multiplier, sorting_method=None, with_global=False)

    ### second layer
    P2 = int(num_point/8)
    C2 = 64
    C_pts_fts = C1 // 4
    depth_multiplier = math.ceil(C2/C1)
    indices = tf_sampling.farthest_point_sample(P2, x_layer1)
    x_layer2 = tf_sampling.gather_point(x_layer1, indices)  # (batch_size, num_point/2, 3)
    fts_xconv_2 = xconv_new(x_layer1, fts_xconv_1, x_layer2, 'xconv_2_', batch_size, K=nsample, radius=0.2, P=P2, C=C2,
                      C_pts_fts=C_pts_fts, is_training=is_training, with_X_transformation=True,
                      depth_multiplier=depth_multiplier, sorting_method=None, with_global=False)

    ### third layer
    P3 = int(num_point/32)
    C3 = 128
    C_pts_fts = C2 // 4
    depth_multiplier = math.ceil(C3 / C2)
    indices = tf_sampling.farthest_point_sample(P3, x_layer2)
    x_layer3 = tf_sampling.gather_point(x_layer2, indices)  # (batch_size, num_point/2, 3)
    fts_xconv_3 = xconv_new(x_layer2, fts_xconv_2, x_layer3, 'xconv_3_', batch_size, K=nsample, radius=0.3, P=P3, C=C3,
                      C_pts_fts=C_pts_fts, is_training=is_training, with_X_transformation=True,
                      depth_multiplier=depth_multiplier, sorting_method=None, with_global=False)

    ### forth layer
    P4 = int(num_point/128)
    C4 = 256
    C_pts_fts = C3 // 4
    depth_multiplier = math.ceil(C4 / C3)
    indices = tf_sampling.farthest_point_sample(P4, x_layer3)
    x_layer4 = tf_sampling.gather_point(x_layer3, indices)  # (batch_size, num_point/2, 3)
    fts_xconv_4 = xconv_new(x_layer3, fts_xconv_3, x_layer4, 'xconv_4_', batch_size, K=nsample, radius=0.4, P=P4, C=C4,
                      C_pts_fts=C_pts_fts, is_training=is_training, with_X_transformation=True,
                      depth_multiplier=depth_multiplier, sorting_method=None, with_global=False)


    fts_deconv_3 = deconv(pts=x_layer4, fts=fts_xconv_4, qrs=x_layer3, qrs_fts=fts_xconv_3, C=256,
                          tag='xdconv4', is_training=is_training)
    fts_deconv_2 = deconv(pts=x_layer3, fts=fts_deconv_3, qrs=x_layer2, qrs_fts=fts_xconv_2, C=256,
                          tag='xdconv3', is_training=is_training)
    fts_deconv_1 = deconv(pts=x_layer2, fts=fts_deconv_2, qrs=x_layer1, qrs_fts=fts_xconv_1, C=128,
                          tag='xdconv2', is_training=is_training)
    per_point_feature = deconv(pts=x_layer1, fts=fts_deconv_1, qrs=x, qrs_fts=None, C=128,
                          tag='xdconv1', is_training=is_training)


    ########## extract per-shape feature ##############
    attention_feature, spatial_weights = channel_spatial_attention(per_point_feature, is_training, bn_decay,
                                                                   name="attention")  # [B, N, 128]

    net = tf.reduce_mean(attention_feature, axis=1)  # (batch_size, 128)

    # l2 normalization
    per_shape_feature = tf.nn.l2_normalize(net, dim=1)

    return attention_feature, per_shape_feature, spatial_weights

def get_loss(net, net_positive, net_negative, cluster_mean, indices, T=0.07):
    """
    net: (batch_size, 128)
    net_positive: (batch_size, 128)
    indices: (batch_size) specify which cluster this object beglongs to
    culster_mean: (cluster_num, 128)
    T: temperature parameter, commonly set 0.07
    """
    ## add dropout
    net_drop = tf.nn.dropout(net, keep_prob=0.8)

    temp = tf.exp((tf.matmul(cluster_mean, tf.transpose(net_drop))) / T)  # (N, batch_size)
    Z = tf.reduce_sum(temp, axis=0)  # (batch_size,)
    vv = tf.gather(cluster_mean, indices=indices)  # (batch_size, 128)
    numerator = tf.exp((tf.matmul(vv, tf.transpose(net_drop))) / T)  # (batch_size, batch_size)
    numerator = tf.diag_part(numerator)  # (batch_size,)
    p = numerator / Z
    cluster_loss = -1.0 * tf.reduce_mean(tf.log(p))

    ## contrastive loss
    dist = tf.reduce_sum(tf.square(net-net_positive), axis=1)
    loss_positive = tf.reduce_mean(dist)

    batch_size = net.get_shape()[0].value
    dist = tf.reduce_sum(tf.square(net - net_negative), axis=1)
    dist = tf.reshape(dist, shape=(batch_size, 1))
    dist_sqrt = tf.sqrt(dist)
    g = tf.constant(2.0, shape=(batch_size, 1))
    mat_0 = tf.zeros((batch_size, 1), dtype=tf.float32)
    temp = g-dist_sqrt
    dist_concat = tf.concat([temp, mat_0], 1)
    dist_max = tf.reduce_max(dist_concat, axis=1)
    loss_negative = tf.reduce_mean(tf.square(dist_max))

    contrastive_loss = loss_positive + loss_negative

    total_loss = cluster_loss + 3.0*contrastive_loss

    return cluster_loss, loss_positive, loss_negative, contrastive_loss, total_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        gt = tf.ones((32, 2048, 3))
        global_feat = get_model(inputs, tf.constant(True))

        print np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

        net = tf.zeros((32, 128))
        net_positive = tf.zeros((32, 128))
        net_negative = tf.zeros((32, 128))
        buffer = tf.zeros((100, 128))
        indices = tf.zeros((32), dtype=tf.int64)
        loss = get_loss(net,net_positive, net_negative,buffer,indices)
