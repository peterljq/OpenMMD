# -*- coding: utf-8 -*-
"""
Created on Jul 13 16:20 2017

@author: Denis Tome'
"""
from . import utils
import cv2
import numpy as np
import tensorflow as tf

import abc
ABC = abc.ABCMeta('ABC', (object,), {})

__all__ = [
    'PoseEstimatorInterface',
    'PoseEstimator'
]


class PoseEstimatorInterface(ABC):

    @abc.abstractmethod
    def initialise(self):
        pass

    @abc.abstractmethod
    def estimate(self, image):
        return

    @abc.abstractmethod
    def close(self):
        pass


class PoseEstimator(PoseEstimatorInterface):

    def __init__(self, image_size, session_path, prob_model_path):
        """Initialising the graph in tensorflow.
        INPUT:
            image_size: Size of the image in the format (w x h x 3)"""

        self.session = None
        self.poseLifting = utils.Prob3dPose(prob_model_path)
        self.sess = -1
        self.orig_img_size = np.array(image_size)
        self.scale = utils.config.INPUT_SIZE / (self.orig_img_size[0] * 1.0)
        self.img_size = np.round(
            self.orig_img_size * self.scale).astype(np.int32)
        self.image_in = None
        self.heatmap_person_large = None
        self.pose_image_in = None
        self.pose_centermap_in = None
        self.heatmap_pose = None
        self.session_path = session_path

    def initialise(self):
        """Load saved model in the graph
        INPUT:
            sess_path: path to the dir containing the tensorflow saved session
        OUTPUT:
            sess: tensorflow session"""

        '''
        TODO: _N shadows built-in name '_N'
        '''
        _N = 16

        tf.reset_default_graph()
        with tf.variable_scope('CPM'):
            # placeholders for person network
            self.image_in = tf.placeholder(
                tf.float32, [1, utils.config.INPUT_SIZE, self.img_size[1], 3])

            heatmap_person = utils.inference_person(self.image_in)

            self.heatmap_person_large = tf.image.resize_images(
                heatmap_person, [utils.config.INPUT_SIZE, self.img_size[1]])

            # placeholders for pose network
            self.pose_image_in = tf.placeholder(
                tf.float32,
                [_N, utils.config.INPUT_SIZE, utils.config.INPUT_SIZE, 3])

            self.pose_centermap_in = tf.placeholder(
                tf.float32,
                [_N, utils.config.INPUT_SIZE, utils.config.INPUT_SIZE, 1])

            self.heatmap_pose = utils.inference_pose(
                self.pose_image_in, self.pose_centermap_in)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, self.session_path)

        self.session = sess

    def estimate(self, image):
        """
        Estimate 2d and 3d poses on the image.
        INPUT:
            image: RGB image in the format (w x h x 3)
            sess: tensorflow session
        OUTPUT:
            pose_2d: 2D pose for each of the people in the image in the format
            (num_ppl x num_joints x 2) visibility: vector containing a bool
            value for each joint representing the visibility of the joint in
            the image (could be due to occlusions or the joint is not in the
            image) pose_3d: 3D pose for each of the people in the image in the
            format (num_ppl x 3 x num_joints)
        """

        sess = self.session

        image = cv2.resize(image, (0, 0), fx=self.scale,
                           fy=self.scale, interpolation=cv2.INTER_CUBIC)
        b_image = np.array(image[np.newaxis] / 255.0 - 0.5, dtype=np.float32)

        hmap_person = sess.run(self.heatmap_person_large, {
                               self.image_in: b_image})

        hmap_person = np.squeeze(hmap_person)
        centers = utils.detect_objects_heatmap(hmap_person)
        b_pose_image, b_pose_cmap = utils.prepare_input_posenet(
            b_image[0], centers,
            [utils.config.INPUT_SIZE, image.shape[1]],
            [utils.config.INPUT_SIZE, utils.config.INPUT_SIZE])

        feed_dict = {
            self.pose_image_in: b_pose_image,
            self.pose_centermap_in: b_pose_cmap
        }
        _hmap_pose = sess.run(self.heatmap_pose, feed_dict)

        # Estimate 2D poses
        estimated_2d_pose, visibility = utils.detect_parts_heatmaps(
            _hmap_pose, centers,
            [utils.config.INPUT_SIZE, utils.config.INPUT_SIZE])

        # Estimate 3D poses
        transformed_pose2d, weights = self.poseLifting.transform_joints(
            estimated_2d_pose.copy(), visibility)
        pose_3d = self.poseLifting.compute_3d(transformed_pose2d, weights)
        pose_2d = np.round(estimated_2d_pose / self.scale).astype(np.int32)

        return pose_2d, visibility, pose_3d

    def close(self):
        self.session.close()
