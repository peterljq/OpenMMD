"""
TODO: Almost all variables in this file violate PEP 8 naming conventions
"""

import tensorflow as tf
import tensorflow.contrib.layers as layers

__all__ = [
    'inference_person',
    'inference_pose'
]


def inference_person(image):
    with tf.variable_scope('PersonNet'):
        conv1_1 = layers.conv2d(
            image, 64, 3, 1, activation_fn=None, scope='conv1_1')
        conv1_1 = tf.nn.relu(conv1_1)
        conv1_2 = layers.conv2d(
            conv1_1, 64, 3, 1, activation_fn=None, scope='conv1_2')
        conv1_2 = tf.nn.relu(conv1_2)
        pool1_stage1 = layers.max_pool2d(conv1_2, 2, 2)
        conv2_1 = layers.conv2d(pool1_stage1, 128, 3, 1,
                                activation_fn=None, scope='conv2_1')
        conv2_1 = tf.nn.relu(conv2_1)
        conv2_2 = layers.conv2d(
            conv2_1, 128, 3, 1, activation_fn=None, scope='conv2_2')
        conv2_2 = tf.nn.relu(conv2_2)
        pool2_stage1 = layers.max_pool2d(conv2_2, 2, 2)
        conv3_1 = layers.conv2d(pool2_stage1, 256, 3, 1,
                                activation_fn=None, scope='conv3_1')
        conv3_1 = tf.nn.relu(conv3_1)
        conv3_2 = layers.conv2d(
            conv3_1, 256, 3, 1, activation_fn=None, scope='conv3_2')
        conv3_2 = tf.nn.relu(conv3_2)
        conv3_3 = layers.conv2d(
            conv3_2, 256, 3, 1, activation_fn=None, scope='conv3_3')
        conv3_3 = tf.nn.relu(conv3_3)
        conv3_4 = layers.conv2d(
            conv3_3, 256, 3, 1, activation_fn=None, scope='conv3_4')
        conv3_4 = tf.nn.relu(conv3_4)
        pool3_stage1 = layers.max_pool2d(conv3_4, 2, 2)
        conv4_1 = layers.conv2d(pool3_stage1, 512, 3, 1,
                                activation_fn=None, scope='conv4_1')
        conv4_1 = tf.nn.relu(conv4_1)
        conv4_2 = layers.conv2d(
            conv4_1, 512, 3, 1, activation_fn=None, scope='conv4_2')
        conv4_2 = tf.nn.relu(conv4_2)
        conv4_3 = layers.conv2d(
            conv4_2, 512, 3, 1, activation_fn=None, scope='conv4_3')
        conv4_3 = tf.nn.relu(conv4_3)
        conv4_4 = layers.conv2d(
            conv4_3, 512, 3, 1, activation_fn=None, scope='conv4_4')
        conv4_4 = tf.nn.relu(conv4_4)
        conv5_1 = layers.conv2d(
            conv4_4, 512, 3, 1, activation_fn=None, scope='conv5_1')
        conv5_1 = tf.nn.relu(conv5_1)
        conv5_2_CPM = layers.conv2d(
            conv5_1, 128, 3, 1, activation_fn=None, scope='conv5_2_CPM')
        conv5_2_CPM = tf.nn.relu(conv5_2_CPM)
        conv6_1_CPM = layers.conv2d(
            conv5_2_CPM, 512, 1, 1, activation_fn=None, scope='conv6_1_CPM')
        conv6_1_CPM = tf.nn.relu(conv6_1_CPM)
        conv6_2_CPM = layers.conv2d(
            conv6_1_CPM, 1, 1, 1, activation_fn=None, scope='conv6_2_CPM')
        concat_stage2 = tf.concat([conv6_2_CPM, conv5_2_CPM], 3)
        Mconv1_stage2 = layers.conv2d(
            concat_stage2, 128, 7, 1, activation_fn=None,
            scope='Mconv1_stage2')
        Mconv1_stage2 = tf.nn.relu(Mconv1_stage2)
        Mconv2_stage2 = layers.conv2d(
            Mconv1_stage2, 128, 7, 1, activation_fn=None,
            scope='Mconv2_stage2')
        Mconv2_stage2 = tf.nn.relu(Mconv2_stage2)
        Mconv3_stage2 = layers.conv2d(
            Mconv2_stage2, 128, 7, 1, activation_fn=None,
            scope='Mconv3_stage2')
        Mconv3_stage2 = tf.nn.relu(Mconv3_stage2)
        Mconv4_stage2 = layers.conv2d(
            Mconv3_stage2, 128, 7, 1, activation_fn=None,
            scope='Mconv4_stage2')
        Mconv4_stage2 = tf.nn.relu(Mconv4_stage2)
        Mconv5_stage2 = layers.conv2d(
            Mconv4_stage2, 128, 7, 1, activation_fn=None,
            scope='Mconv5_stage2')
        Mconv5_stage2 = tf.nn.relu(Mconv5_stage2)
        Mconv6_stage2 = layers.conv2d(
            Mconv5_stage2, 128, 1, 1, activation_fn=None,
            scope='Mconv6_stage2')
        Mconv6_stage2 = tf.nn.relu(Mconv6_stage2)
        Mconv7_stage2 = layers.conv2d(
            Mconv6_stage2, 1, 1, 1, activation_fn=None, scope='Mconv7_stage2')
        concat_stage3 = tf.concat([Mconv7_stage2, conv5_2_CPM], 3)
        Mconv1_stage3 = layers.conv2d(
            concat_stage3, 128, 7, 1, activation_fn=None,
            scope='Mconv1_stage3')
        Mconv1_stage3 = tf.nn.relu(Mconv1_stage3)
        Mconv2_stage3 = layers.conv2d(
            Mconv1_stage3, 128, 7, 1, activation_fn=None,
            scope='Mconv2_stage3')
        Mconv2_stage3 = tf.nn.relu(Mconv2_stage3)
        Mconv3_stage3 = layers.conv2d(
            Mconv2_stage3, 128, 7, 1, activation_fn=None,
            scope='Mconv3_stage3')
        Mconv3_stage3 = tf.nn.relu(Mconv3_stage3)
        Mconv4_stage3 = layers.conv2d(
            Mconv3_stage3, 128, 7, 1, activation_fn=None,
            scope='Mconv4_stage3')
        Mconv4_stage3 = tf.nn.relu(Mconv4_stage3)
        Mconv5_stage3 = layers.conv2d(
            Mconv4_stage3, 128, 7, 1, activation_fn=None,
            scope='Mconv5_stage3')
        Mconv5_stage3 = tf.nn.relu(Mconv5_stage3)
        Mconv6_stage3 = layers.conv2d(
            Mconv5_stage3, 128, 1, 1, activation_fn=None,
            scope='Mconv6_stage3')
        Mconv6_stage3 = tf.nn.relu(Mconv6_stage3)
        Mconv7_stage3 = layers.conv2d(
            Mconv6_stage3, 1, 1, 1, activation_fn=None,
            scope='Mconv7_stage3')
        concat_stage4 = tf.concat([Mconv7_stage3, conv5_2_CPM], 3)
        Mconv1_stage4 = layers.conv2d(
            concat_stage4, 128, 7, 1, activation_fn=None,
            scope='Mconv1_stage4')
        Mconv1_stage4 = tf.nn.relu(Mconv1_stage4)
        Mconv2_stage4 = layers.conv2d(
            Mconv1_stage4, 128, 7, 1, activation_fn=None,
            scope='Mconv2_stage4')
        Mconv2_stage4 = tf.nn.relu(Mconv2_stage4)
        Mconv3_stage4 = layers.conv2d(
            Mconv2_stage4, 128, 7, 1, activation_fn=None,
            scope='Mconv3_stage4')
        Mconv3_stage4 = tf.nn.relu(Mconv3_stage4)
        Mconv4_stage4 = layers.conv2d(
            Mconv3_stage4, 128, 7, 1, activation_fn=None,
            scope='Mconv4_stage4')
        Mconv4_stage4 = tf.nn.relu(Mconv4_stage4)
        Mconv5_stage4 = layers.conv2d(
            Mconv4_stage4, 128, 7, 1, activation_fn=None,
            scope='Mconv5_stage4')
        Mconv5_stage4 = tf.nn.relu(Mconv5_stage4)
        Mconv6_stage4 = layers.conv2d(
            Mconv5_stage4, 128, 1, 1, activation_fn=None,
            scope='Mconv6_stage4')
        Mconv6_stage4 = tf.nn.relu(Mconv6_stage4)
        Mconv7_stage4 = layers.conv2d(
            Mconv6_stage4, 1, 1, 1, activation_fn=None, scope='Mconv7_stage4')
    return Mconv7_stage4


def inference_pose(image, center_map):
    with tf.variable_scope('PoseNet'):
        pool_center_lower = layers.avg_pool2d(center_map, 9, 8, padding='SAME')
        conv1_1 = layers.conv2d(
            image, 64, 3, 1, activation_fn=None, scope='conv1_1')
        conv1_1 = tf.nn.relu(conv1_1)
        conv1_2 = layers.conv2d(
            conv1_1, 64, 3, 1, activation_fn=None, scope='conv1_2')
        conv1_2 = tf.nn.relu(conv1_2)
        pool1_stage1 = layers.max_pool2d(conv1_2, 2, 2)
        conv2_1 = layers.conv2d(pool1_stage1, 128, 3, 1,
                                activation_fn=None, scope='conv2_1')
        conv2_1 = tf.nn.relu(conv2_1)
        conv2_2 = layers.conv2d(
            conv2_1, 128, 3, 1, activation_fn=None, scope='conv2_2')
        conv2_2 = tf.nn.relu(conv2_2)
        pool2_stage1 = layers.max_pool2d(conv2_2, 2, 2)
        conv3_1 = layers.conv2d(pool2_stage1, 256, 3, 1,
                                activation_fn=None, scope='conv3_1')
        conv3_1 = tf.nn.relu(conv3_1)
        conv3_2 = layers.conv2d(
            conv3_1, 256, 3, 1, activation_fn=None, scope='conv3_2')
        conv3_2 = tf.nn.relu(conv3_2)
        conv3_3 = layers.conv2d(
            conv3_2, 256, 3, 1, activation_fn=None, scope='conv3_3')
        conv3_3 = tf.nn.relu(conv3_3)
        conv3_4 = layers.conv2d(
            conv3_3, 256, 3, 1, activation_fn=None, scope='conv3_4')
        conv3_4 = tf.nn.relu(conv3_4)
        pool3_stage1 = layers.max_pool2d(conv3_4, 2, 2)
        conv4_1 = layers.conv2d(pool3_stage1, 512, 3, 1,
                                activation_fn=None, scope='conv4_1')
        conv4_1 = tf.nn.relu(conv4_1)
        conv4_2 = layers.conv2d(
            conv4_1, 512, 3, 1, activation_fn=None, scope='conv4_2')
        conv4_2 = tf.nn.relu(conv4_2)
        conv4_3_CPM = layers.conv2d(
            conv4_2, 256, 3, 1, activation_fn=None, scope='conv4_3_CPM')
        conv4_3_CPM = tf.nn.relu(conv4_3_CPM)
        conv4_4_CPM = layers.conv2d(
            conv4_3_CPM, 256, 3, 1, activation_fn=None, scope='conv4_4_CPM')
        conv4_4_CPM = tf.nn.relu(conv4_4_CPM)
        conv4_5_CPM = layers.conv2d(
            conv4_4_CPM, 256, 3, 1, activation_fn=None, scope='conv4_5_CPM')
        conv4_5_CPM = tf.nn.relu(conv4_5_CPM)
        conv4_6_CPM = layers.conv2d(
            conv4_5_CPM, 256, 3, 1, activation_fn=None, scope='conv4_6_CPM')
        conv4_6_CPM = tf.nn.relu(conv4_6_CPM)
        conv4_7_CPM = layers.conv2d(
            conv4_6_CPM, 128, 3, 1, activation_fn=None, scope='conv4_7_CPM')
        conv4_7_CPM = tf.nn.relu(conv4_7_CPM)
        conv5_1_CPM = layers.conv2d(
            conv4_7_CPM, 512, 1, 1, activation_fn=None, scope='conv5_1_CPM')
        conv5_1_CPM = tf.nn.relu(conv5_1_CPM)
        conv5_2_CPM = layers.conv2d(
            conv5_1_CPM, 15, 1, 1, activation_fn=None, scope='conv5_2_CPM')
        concat_stage2 = tf.concat(
            [conv5_2_CPM, conv4_7_CPM, pool_center_lower], 3)
        Mconv1_stage2 = layers.conv2d(
            concat_stage2, 128, 7, 1, activation_fn=None,
            scope='Mconv1_stage2')
        Mconv1_stage2 = tf.nn.relu(Mconv1_stage2)
        Mconv2_stage2 = layers.conv2d(
            Mconv1_stage2, 128, 7, 1, activation_fn=None,
            scope='Mconv2_stage2')
        Mconv2_stage2 = tf.nn.relu(Mconv2_stage2)
        Mconv3_stage2 = layers.conv2d(
            Mconv2_stage2, 128, 7, 1, activation_fn=None,
            scope='Mconv3_stage2')
        Mconv3_stage2 = tf.nn.relu(Mconv3_stage2)
        Mconv4_stage2 = layers.conv2d(
            Mconv3_stage2, 128, 7, 1, activation_fn=None,
            scope='Mconv4_stage2')
        Mconv4_stage2 = tf.nn.relu(Mconv4_stage2)
        Mconv5_stage2 = layers.conv2d(
            Mconv4_stage2, 128, 7, 1, activation_fn=None,
            scope='Mconv5_stage2')
        Mconv5_stage2 = tf.nn.relu(Mconv5_stage2)
        Mconv6_stage2 = layers.conv2d(
            Mconv5_stage2, 128, 1, 1, activation_fn=None,
            scope='Mconv6_stage2')
        Mconv6_stage2 = tf.nn.relu(Mconv6_stage2)
        Mconv7_stage2 = layers.conv2d(
            Mconv6_stage2, 15, 1, 1, activation_fn=None, scope='Mconv7_stage2')
        concat_stage3 = tf.concat(
            [Mconv7_stage2, conv4_7_CPM, pool_center_lower], 3)
        Mconv1_stage3 = layers.conv2d(
            concat_stage3, 128, 7, 1, activation_fn=None,
            scope='Mconv1_stage3')
        Mconv1_stage3 = tf.nn.relu(Mconv1_stage3)
        Mconv2_stage3 = layers.conv2d(
            Mconv1_stage3, 128, 7, 1, activation_fn=None,
            scope='Mconv2_stage3')
        Mconv2_stage3 = tf.nn.relu(Mconv2_stage3)
        Mconv3_stage3 = layers.conv2d(
            Mconv2_stage3, 128, 7, 1, activation_fn=None,
            scope='Mconv3_stage3')
        Mconv3_stage3 = tf.nn.relu(Mconv3_stage3)
        Mconv4_stage3 = layers.conv2d(
            Mconv3_stage3, 128, 7, 1, activation_fn=None,
            scope='Mconv4_stage3')
        Mconv4_stage3 = tf.nn.relu(Mconv4_stage3)
        Mconv5_stage3 = layers.conv2d(
            Mconv4_stage3, 128, 7, 1, activation_fn=None,
            scope='Mconv5_stage3')
        Mconv5_stage3 = tf.nn.relu(Mconv5_stage3)
        Mconv6_stage3 = layers.conv2d(
            Mconv5_stage3, 128, 1, 1, activation_fn=None,
            scope='Mconv6_stage3')
        Mconv6_stage3 = tf.nn.relu(Mconv6_stage3)
        Mconv7_stage3 = layers.conv2d(
            Mconv6_stage3, 15, 1, 1, activation_fn=None, scope='Mconv7_stage3')
        concat_stage4 = tf.concat(
            [Mconv7_stage3, conv4_7_CPM, pool_center_lower], 3)
        Mconv1_stage4 = layers.conv2d(
            concat_stage4, 128, 7, 1, activation_fn=None,
            scope='Mconv1_stage4')
        Mconv1_stage4 = tf.nn.relu(Mconv1_stage4)
        Mconv2_stage4 = layers.conv2d(
            Mconv1_stage4, 128, 7, 1, activation_fn=None,
            scope='Mconv2_stage4')
        Mconv2_stage4 = tf.nn.relu(Mconv2_stage4)
        Mconv3_stage4 = layers.conv2d(
            Mconv2_stage4, 128, 7, 1, activation_fn=None,
            scope='Mconv3_stage4')
        Mconv3_stage4 = tf.nn.relu(Mconv3_stage4)
        Mconv4_stage4 = layers.conv2d(
            Mconv3_stage4, 128, 7, 1, activation_fn=None,
            scope='Mconv4_stage4')
        Mconv4_stage4 = tf.nn.relu(Mconv4_stage4)
        Mconv5_stage4 = layers.conv2d(
            Mconv4_stage4, 128, 7, 1, activation_fn=None,
            scope='Mconv5_stage4')
        Mconv5_stage4 = tf.nn.relu(Mconv5_stage4)
        Mconv6_stage4 = layers.conv2d(
            Mconv5_stage4, 128, 1, 1, activation_fn=None,
            scope='Mconv6_stage4')
        Mconv6_stage4 = tf.nn.relu(Mconv6_stage4)
        Mconv7_stage4 = layers.conv2d(
            Mconv6_stage4, 15, 1, 1, activation_fn=None, scope='Mconv7_stage4')
        concat_stage5 = tf.concat(
            [Mconv7_stage4, conv4_7_CPM, pool_center_lower], 3)
        Mconv1_stage5 = layers.conv2d(
            concat_stage5, 128, 7, 1, activation_fn=None,
            scope='Mconv1_stage5')
        Mconv1_stage5 = tf.nn.relu(Mconv1_stage5)
        Mconv2_stage5 = layers.conv2d(
            Mconv1_stage5, 128, 7, 1, activation_fn=None,
            scope='Mconv2_stage5')
        Mconv2_stage5 = tf.nn.relu(Mconv2_stage5)
        Mconv3_stage5 = layers.conv2d(
            Mconv2_stage5, 128, 7, 1, activation_fn=None,
            scope='Mconv3_stage5')
        Mconv3_stage5 = tf.nn.relu(Mconv3_stage5)
        Mconv4_stage5 = layers.conv2d(
            Mconv3_stage5, 128, 7, 1, activation_fn=None,
            scope='Mconv4_stage5')
        Mconv4_stage5 = tf.nn.relu(Mconv4_stage5)
        Mconv5_stage5 = layers.conv2d(
            Mconv4_stage5, 128, 7, 1, activation_fn=None,
            scope='Mconv5_stage5')
        Mconv5_stage5 = tf.nn.relu(Mconv5_stage5)
        Mconv6_stage5 = layers.conv2d(
            Mconv5_stage5, 128, 1, 1, activation_fn=None,
            scope='Mconv6_stage5')
        Mconv6_stage5 = tf.nn.relu(Mconv6_stage5)
        Mconv7_stage5 = layers.conv2d(
            Mconv6_stage5, 15, 1, 1, activation_fn=None, scope='Mconv7_stage5')
        concat_stage6 = tf.concat(
            [Mconv7_stage5, conv4_7_CPM, pool_center_lower], 3)
        Mconv1_stage6 = layers.conv2d(
            concat_stage6, 128, 7, 1, activation_fn=None,
            scope='Mconv1_stage6')
        Mconv1_stage6 = tf.nn.relu(Mconv1_stage6)
        Mconv2_stage6 = layers.conv2d(
            Mconv1_stage6, 128, 7, 1, activation_fn=None,
            scope='Mconv2_stage6')
        Mconv2_stage6 = tf.nn.relu(Mconv2_stage6)
        Mconv3_stage6 = layers.conv2d(
            Mconv2_stage6, 128, 7, 1, activation_fn=None,
            scope='Mconv3_stage6')
        Mconv3_stage6 = tf.nn.relu(Mconv3_stage6)
        Mconv4_stage6 = layers.conv2d(
            Mconv3_stage6, 128, 7, 1, activation_fn=None,
            scope='Mconv4_stage6')
        Mconv4_stage6 = tf.nn.relu(Mconv4_stage6)
        Mconv5_stage6 = layers.conv2d(
            Mconv4_stage6, 128, 7, 1, activation_fn=None,
            scope='Mconv5_stage6')
        Mconv5_stage6 = tf.nn.relu(Mconv5_stage6)
        Mconv6_stage6 = layers.conv2d(
            Mconv5_stage6, 128, 1, 1, activation_fn=None,
            scope='Mconv6_stage6')
        Mconv6_stage6 = tf.nn.relu(Mconv6_stage6)
        Mconv7_stage6 = layers.conv2d(
            Mconv6_stage6, 15, 1, 1, activation_fn=None,
            scope='Mconv7_stage6')
    return Mconv7_stage6
