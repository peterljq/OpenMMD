import sys

import os
import numpy as np
import cv2
import chainer
import chainer.functions as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import projection_gan


class JointsForPCK(object):
    """
    @see Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision
    """
    from_h36m_joints = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]


def load_model(opts):
    model = projection_gan.pose.posenet.MLP(mode='generator',
        use_bn=opts['use_bn'], activate_func=getattr(F, opts['activate_func']))
    return model


def color_jet(x):
    if x < 0.25:
        b = 255
        g = x / 0.25 * 255
        r = 0
    elif x >= 0.25 and x < 0.5:
        b = 255 - (x - 0.25) / 0.25 * 255
        g = 255
        r = 0
    elif x >= 0.5 and x < 0.75:
        b = 0
        g = 255
        r = (x - 0.5) / 0.25 * 255
    else:
        b = 0
        g = 255 - (x - 0.75) / 0.25 * 255
        r = 255
    return int(b), int(g), int(r)


def create_projection_img(array, theta):
    x = array[:, 0::3]
    y = array[:, 1::3]
    z = array[:, 2::3]

    xx = x * np.cos(theta) + z * np.sin(theta)
    fake = np.stack((xx, y), axis=-1).flatten()
    return create_img(fake)


def create_img(arr, img=None):
    ps = [0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
    qs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    xs = arr[0::2].copy()
    ys = arr[1::2].copy()
    if img is None:
        xs *= 80
        xs += 100
        ys *= 80
        ys += 150
        xs = xs.astype('i')
        ys = ys.astype('i')
        img = np.zeros((350, 200, 3), dtype=np.uint8) + 160
        img = cv2.line(img, (100, 0), (100, 350), (255, 255, 255), 1)
        img = cv2.line(img, (0, 150), (200, 150), (255, 255, 255), 1)
        img = cv2.rectangle(img, (0, 0), (200, 350), (255, 255, 255), 3)
    for i, (p, q) in enumerate(zip(ps, qs)):
        c = 1 / (len(ps) - 1) * i
        b, g, r = color_jet(c)
        img = cv2.line(img, (xs[p], ys[p]), (xs[q], ys[q]), (b, g, r), 2)
    for i in range(17):
        c = 1 / 16 * i
        b, g, r = color_jet(c)
        img = cv2.circle(img, (xs[i], ys[i]), 3, (b, g, r), 3)
    return img
