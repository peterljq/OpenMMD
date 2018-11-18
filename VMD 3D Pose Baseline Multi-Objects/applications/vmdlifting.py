#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# vmdlifting.py - estimate 3D pose by "Lifting-from-the-Deep", and convert the pose data to VMD
#
# This program is derived from demo.py in Lifting-from-the-Deep which is created by Denis Tome'

from __future__ import print_function

def usage(prog):
    print('usage: ' + prog + ' IMAGE_FILE VMD_FILE [POSITION_FILE]')
    sys.exit()

import __init__

from lifting import PoseEstimator
from lifting.utils import draw_limbs
from lifting.utils import plot_pose

import cv2
import matplotlib.pyplot as plt
from os.path import dirname, realpath
from pos2vmd import pos2vmd
from head_face import head_face_estimation

DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'

def vmdlifting(image_file, vmd_file, position_file):
    image_file_path = realpath(DIR_PATH) + '/' + image_file
    image = cv2.imread(image_file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb

    # create pose estimator
    image_size = image.shape

    pose_estimator = PoseEstimator(image_size, SESSION_PATH, PROB_MODEL_PATH)

    # load model
    pose_estimator.initialise()

    # estimation
    pose_2d, visibility, pose_3d = pose_estimator.estimate(image)

    # close model
    pose_estimator.close()

    if (position_file is not None):
        # dump 3d joint position data to position_file
        fout = open(position_file, "w")
        for pose in pose_3d:
            for j in range(pose.shape[1]):
                print(j, pose[0, j], pose[1, j], pose[2, j], file=fout)
        fout.close()

    # head position & face expression
    head_rotation, expression_frames = head_face_estimation(image_file_path)
    pos2vmd(pose_3d, vmd_file, head_rotation, expression_frames)

    # Show 2D and 3D poses
    # display_results(image, pose_2d, visibility, pose_3d)


def display_results(in_image, data_2d, joint_visibility, data_3d):
    """Plot 2D and 3D poses for each of the people in the image."""
    plt.figure()
    draw_limbs(in_image, data_2d, joint_visibility)
    plt.imshow(in_image)
    plt.axis('off')

    # Show 3D poses
    for single_3D in data_3d:
        # or plot_pose(Prob3dPose.centre_all(single_3D))
        plot_pose(single_3D)

    plt.show()

if __name__ == '__main__':
    import sys
    if (len(sys.argv) < 3):
        usage(sys.argv[0])
        
    image_file = sys.argv[1]
    vmd_file = sys.argv[2]
    dump_file = None
    if (len(sys.argv) >3 ):
        dump_file = sys.argv[3]

    vmdlifting(image_file, vmd_file, dump_file)
