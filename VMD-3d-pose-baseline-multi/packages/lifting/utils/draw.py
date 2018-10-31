# -*- coding: utf-8 -*-
"""
Created on Mar 23 15:04 2017

@author: Denis Tome'
"""
import cv2
import numpy as np
from .config import JOINT_DRAW_SIZE
from .config import LIMB_DRAW_SIZE
from .config import NORMALISATION_COEFFICIENT
import matplotlib.pyplot as plt
import math

__all__ = [
    'draw_limbs',
    'plot_pose'
]


def draw_limbs(image, pose_2d, visible):
    """Draw the 2D pose without the occluded/not visible joints."""

    _COLORS = [
        [0, 0, 255], [0, 170, 255], [0, 255, 170], [0, 255, 0],
        [170, 255, 0], [255, 170, 0], [255, 0, 0], [255, 0, 170],
        [170, 0, 255]
    ]
    _LIMBS = np.array([0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9,
                       9, 10, 11, 12, 12, 13]).reshape((-1, 2))

    _NORMALISATION_FACTOR = int(math.floor(math.sqrt(image.shape[0] * image.shape[1] / NORMALISATION_COEFFICIENT)))

    for oid in range(pose_2d.shape[0]):
        for lid, (p0, p1) in enumerate(_LIMBS):
            if not (visible[oid][p0] and visible[oid][p1]):
                continue
            y0, x0 = pose_2d[oid][p0]
            y1, x1 = pose_2d[oid][p1]
            cv2.circle(image, (x0, y0), JOINT_DRAW_SIZE *_NORMALISATION_FACTOR , _COLORS[lid], -1)
            cv2.circle(image, (x1, y1), JOINT_DRAW_SIZE*_NORMALISATION_FACTOR , _COLORS[lid], -1)
            cv2.line(image, (x0, y0), (x1, y1),
                     _COLORS[lid], LIMB_DRAW_SIZE*_NORMALISATION_FACTOR , 16)


def plot_pose(pose):
    """Plot the 3D pose showing the joint connections."""
    import mpl_toolkits.mplot3d.axes3d as p3

    _CONNECTION = [
        [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
        [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15],
        [15, 16]]

    def joint_color(j):
        """
        TODO: 'j' shadows name 'j' from outer scope
        """

        colors = [(0, 0, 0), (255, 0, 255), (0, 0, 255),
                  (0, 255, 255), (255, 0, 0), (0, 255, 0)]
        _c = 0
        if j in range(1, 4):
            _c = 1
        if j in range(4, 7):
            _c = 2
        if j in range(9, 11):
            _c = 3
        if j in range(11, 14):
            _c = 4
        if j in range(14, 17):
            _c = 5
        return colors[_c]

    assert (pose.ndim == 2)
    assert (pose.shape[0] == 3)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for c in _CONNECTION:
        col = '#%02x%02x%02x' % joint_color(c[0])
        ax.plot([pose[0, c[0]], pose[0, c[1]]],
                [pose[1, c[0]], pose[1, c[1]]],
                [pose[2, c[0]], pose[2, c[1]]], c=col)
    for j in range(pose.shape[1]):
        col = '#%02x%02x%02x' % joint_color(j)
        ax.scatter(pose[0, j], pose[1, j], pose[2, j],
                   c=col, marker='o', edgecolor=col)
    smallest = pose.min()
    largest = pose.max()
    ax.set_xlim3d(smallest, largest)
    ax.set_ylim3d(smallest, largest)
    ax.set_zlim3d(smallest, largest)

    return fig


