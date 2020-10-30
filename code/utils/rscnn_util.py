import tensorflow as tf
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'sampling'))
from tf_grouping import query_ball_point, group_point, knn_point
from tf_sampling import farthest_point_sample, gather_point
import tf_util

def sample_and_group(sample_pt_num, radius, neighbor_size, input_xyz, input_features):
    '''
    Input:
        sample_pt_num: how many points to keep
        radius: query ball radius
        neighbor_size: how many neighbor points
        input_xyz: (batch_size, npoints, 3)
        input_features: (batch_size, npoint, C)
    Output:
        sampled_xyz: (batch_size, sample_pt_num, 3)
        idx: (batch_size, sample_pt_num, neighbor_size)
        sampled_grouped_relation: (batch_size, sample_pt_num, neighbor_size, 10)
        sampled_grouped_features: (batch_size, sample_pt_num, neighbor_size, C)

    '''

    sampled_xyz = gather_point(input_xyz, farthest_point_sample(sample_pt_num, input_xyz))  # (batch_size, sample_pt_num, 3)
    idx, pts_cnt = query_ball_point(radius, neighbor_size, input_xyz, sampled_xyz)
    sampled_grouped_xyz = group_point(input_xyz, idx)  # (batch_size, sample_pt_num, neighbor_size, 3)
    sampled_grouped_features = group_point(input_features, idx)

    sampled_center_xyz = tf.tile(tf.expand_dims(sampled_xyz, 2), [1, 1, neighbor_size, 1])  # (batch_size, npoint, nsample, 3)

    euclidean = tf.reduce_sum(tf.square(sampled_grouped_xyz-sampled_center_xyz), axis=-1, keepdims=True)  # (batch_size, npoint, nsample, 1)
    sampled_grouped_relation = tf.concat([euclidean, sampled_center_xyz-sampled_grouped_xyz,
                                  sampled_center_xyz, sampled_grouped_xyz], axis=-1)  # (batch_size, npoint, nsample, 10)

    return sampled_xyz, idx, sampled_grouped_relation, sampled_grouped_features

def rs_cnn(xyz_feature, xyz, npoint, radius, nsample, mlp, is_training, bn_decay, scope, bn=True):

    # xyz_feature: (B, N, C)

    sampled_xyz, idx, \
    sampled_grouped_relation, sampled_grouped_features = sample_and_group(npoint, radius,
                                                                          nsample, xyz, xyz_feature)

    with tf.variable_scope(scope) as sc:
        for i, num_out_channel in enumerate(mlp):
            sampled_grouped_relation = tf_util.conv2d(sampled_grouped_relation, num_out_channel, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d' % (i), bn_decay=bn_decay,
                                        data_format='NHWC')      # (B, sample_pt_num, neighbor_size, C)

        rscnn_feature = tf.reduce_max(sampled_grouped_relation*sampled_grouped_features, axis=2)  # [B, N, C]
        # rscnn_feature = tf.nn.relu(rscnn_feature)

    return rscnn_feature, idx, sampled_xyz

def rs_cnn_v1(xyz_feature, xyz, npoint, radius, nsample, mlp, is_training, bn_decay, scope, bn=True):

    # xyz_feature: (B, N, C)

    sampled_xyz, idx, \
    sampled_grouped_relation, sampled_grouped_features = sample_and_group(npoint, radius,
                                                                          nsample, xyz, xyz_feature)

    with tf.variable_scope(scope) as sc:
        for i, num_out_channel in enumerate(mlp):
            sampled_grouped_relation = tf_util.conv2d(sampled_grouped_relation, num_out_channel, [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d' % (i), bn_decay=bn_decay,
                                        data_format='NHWC')      # (B, sample_pt_num, neighbor_size, C)

        rscnn_feature = tf.concat([sampled_grouped_relation, sampled_grouped_features], axis=-1)
        rscnn_feature = tf_util.conv2d(rscnn_feature, mlp[-1], [1, 1],
                                        padding='VALID', stride=[1, 1],
                                        bn=bn, is_training=is_training,
                                        scope='conv_mix', bn_decay=bn_decay,
                                        data_format='NHWC')
        rscnn_feature = tf.reduce_max(rscnn_feature, axis=2)


    return rscnn_feature, idx, sampled_xyz

