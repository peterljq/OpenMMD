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
from pos2vmd_multi import pos2vmd_multi
from head_face import head_face_estimation

DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'

def vmdlifting_multi(video_file, vmd_file, position_file):
    video_file_path = realpath(video_file)
    
    cap = cv2.VideoCapture(video_file_path)
    
    pose_3d_list = []
    head_rotation_list = []
    expression_frames_list = []
    idx = 0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # 読み込みがなければ終了
        if ret == False:
            break

        print("frame load idx={0}".format(idx))

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # conversion to rgb

        # 念のため、フレーム画像出力
        image_file_path = "{0}/frame_{1:012d}.png".format(dirname(video_file_path), idx)
        cv2.imwrite(image_file_path,image)
        
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
        head_rotation_list.append(head_rotation)
        expression_frames_list.append(expression_frames)

        pose_3d_list.append(pose_3d)

        idx += 1

    # When everything done, release the capture
    cap.release()

    pos2vmd_multi(pose_3d_list, vmd_file, head_rotation_list, expression_frames_list)

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
        
    video_file = sys.argv[1]
    vmd_file = sys.argv[2]
    dump_file = None
    if (len(sys.argv) >3 ):
        dump_file = sys.argv[3]

    vmdlifting_multi(video_file, vmd_file, dump_file)
