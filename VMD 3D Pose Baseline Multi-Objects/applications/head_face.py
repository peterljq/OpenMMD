#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# head_face.py - estimate head pose & facial expression

from __future__ import print_function

def usage(prog):
    print('usage: ' + prog + ' IMAGE_FILE')
    sys.exit()

#   You can get the trained model file from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
#   Note that the license for the iBUG 300-W dataset excludes commercial use.
#   So you should contact Imperial College London to find out if it's OK for
#   you to use this model file in a commercial product.
DEFAULT_PREDICTOR_PATH = 'predictor/shape_predictor_68_face_landmarks.dat'

import cv2
import dlib
import numpy as np
import os
from skimage import io
from PyQt5.QtGui import QQuaternion, QVector3D, QMatrix3x3

# dlib の python_examples/face_landmark_detection.py を改造
def face_landmark_detection(image_path, predictor_path):
    shape_list = []
    image = io.imread(image_path)
    if not os.path.exists(predictor_path):
        print("A trained model for face landmark detection is not found.")
        print("You can get the trained model from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return shape_list
    predictor = dlib.shape_predictor(predictor_path)
    detector = dlib.get_frontal_face_detector()
    dets = detector(image, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(image, d)
        print("Nose tip: {}, Chin: {}".format(shape.part(30), shape.part(8)))
        print("Left eye: {}, Right eye: {}".format(shape.part(45), shape.part(36)))
        print("Mouth-left: {}, Mouth-right: {} ...".format(shape.part(54), shape.part(48)))
        shape_list.append(shape)
    # win = dlib.image_window()
    # win.clear_overlay()
    # win.set_image(image)
    # for shape in shape_list:
    #    win.add_overlay(shape)
    # dlib.hit_enter_to_continue()
    return shape_list

def head_pose_estimation(image_path, shape):
    image = cv2.imread(image_path)
    pos2d_array = []
    for k in [30, 8, 45, 36, 54, 48]:
        pos2d_array.append((shape.part(k).x, shape.part(k).y))
    pos2d = np.array(pos2d_array, dtype = "double")
    pos3d_ini = np.array([(0.0, 0.0, 0.0), (0.0, -6.6, -1.3), (-4.5, 3.4, -2.7),
                          (4.5, 3.4, -2.7), (-3.0, -3.0, -2.5), (3.0, -3.0, -2.5)])
    focal_length = max([image.shape[0], image.shape[1]])
    print("focal_length: ", focal_length)
    camera = np.array([[focal_length, 0, image.shape[1] / 2],
                       [0, focal_length, image.shape[0] / 2],
                       [0, 0, 1]], dtype = "double")
    distortion = np.zeros((4, 1))
    retval, rot_vec, trans_vec = cv2.solvePnP(pos3d_ini, pos2d, camera, distortion,
                                              flags = cv2.SOLVEPNP_ITERATIVE)
    # for debug
    #print("Rot_Vec: \n ", rot_vec)
    #print(type(rot_vec))
    #print(rot_vec.shape)
    #print("Trans_Vec:\n ", trans_vec)
    # 顔の回転を求める
    rot_mat = cv2.Rodrigues(rot_vec)[0]
    proj_mat = np.array([[rot_mat[0][0], rot_mat[0][1], rot_mat[0][2], 0],
                         [rot_mat[1][0], rot_mat[1][1], rot_mat[1][2], 0],
                         [rot_mat[2][0], rot_mat[2][1], rot_mat[2][2], 0]], dtype="double")
    eulerAngles = cv2.decomposeProjectionMatrix(proj_mat)[6]
    print("eulerAngles: \n", eulerAngles)
    
    #head_rotation = QQuaternion.fromEulerAngles(eulerAngles[0], eulerAngles[1], eulerAngles[2])
    head_rotation = QQuaternion.fromEulerAngles(-eulerAngles[0], eulerAngles[1], -eulerAngles[2])
    print("head_rotation: ", head_rotation)
    return head_rotation


def make_expression_frames(shape):
    return None


def head_face_estimation(image_path, predictor_path=None):
    if predictor_path is None:
        predictor_path = DEFAULT_PREDICTOR_PATH
    shape_list = face_landmark_detection(image_path, predictor_path)
    if len(shape_list) == 0:
        return None, None
    head_rotation = head_pose_estimation(image_path, shape_list[0])
    expression_frames = make_expression_frames(shape_list[0])
    return head_rotation, expression_frames


if __name__ == '__main__':
    import sys
    if (len(sys.argv) < 2):
        usage(sys.argv[0])

    head_rotation, expression_frames = head_face_estimation(sys.argv[1])

