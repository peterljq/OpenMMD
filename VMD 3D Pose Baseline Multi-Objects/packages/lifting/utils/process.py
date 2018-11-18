# -*- coding: utf-8 -*-
"""
Created on Mar 23 15:29 2017

@author: Denis Tome'
"""
from __future__ import division

import os
import json
import numpy as np
from lifting.utils import config
import cv2
import skimage.io
import skimage.transform
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from itertools import compress
from scipy.stats import multivariate_normal

__all__ = [
    'detect_objects_heatmap',
    'detect_objects_heatmap',
    'gaussian_kernel',
    'gaussian_heatmap',
    'prepare_input_posenet',
    'detect_parts_heatmaps',
    'import_json',
    'generate_labels',
    'generate_center_map',
    'rescale',
    'crop_image'
]


def detect_objects_heatmap(heatmap):
    data = 256 * heatmap
    data_max = filters.maximum_filter(data, 3)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, 3)
    diff = ((data_max - data_min) > 0.3)
    maxima[diff == 0] = 0
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    objects = np.zeros((num_objects, 2), dtype=np.int32)
    pidx = 0
    for (dy, dx) in slices:
        pos = [(dy.start + dy.stop - 1) // 2, (dx.start + dx.stop - 1) // 2]
        if heatmap[pos[0], pos[1]] > config.CENTER_TR:
            objects[pidx, :] = pos
            pidx += 1
    return objects[:pidx]


def gaussian_kernel(h, w, sigma_h, sigma_w):
    yx = np.mgrid[-h // 2:h // 2, -w // 2:w // 2] ** 2
    return np.exp(-yx[0, :, :] / sigma_h ** 2 - yx[1, :, :] / sigma_w ** 2)


def gaussian_heatmap(h, w, pos_x, pos_y, sigma_h=1, sigma_w=1, init=None):
    """
    Compute the heat-map of size (w x h) with a gaussian distribution fit in
    position (pos_x, pos_y) and a convariance matix defined by the related
    sigma values.
    The resulting heat-map can be summed to a given heat-map init.
    """
    init = init if init is not None else []

    cov_matrix = np.eye(2) * ([sigma_h**2, sigma_w**2])

    x, y = np.mgrid[0:h, 0:w]
    pos = np.dstack((x, y))
    rv = multivariate_normal([pos_x, pos_y], cov_matrix)

    tmp = rv.pdf(pos)
    hmap = np.multiply(
        tmp, np.sqrt(np.power(2 * np.pi, 2) * np.linalg.det(cov_matrix))
    )
    idx = np.where(hmap.flatten() <= np.exp(-4.6052))
    hmap.flatten()[idx] = 0

    if np.size(init) == 0:
        return hmap

    assert (np.shape(init) == hmap.shape)
    hmap += init
    idx = np.where(hmap.flatten() > 1)
    hmap.flatten()[idx] = 1
    return hmap


def prepare_input_posenet(image, objects, size_person, size, sigma=25,
                          max_num_objects=16, border=400):
    result = np.zeros((max_num_objects, size[0], size[1], 4))
    padded_image = np.zeros(
        (1, size_person[0] + border, size_person[1] + border, 4))
    padded_image[0, border // 2:-border // 2,
                 border // 2:-border // 2, :3] = image
    assert len(objects) < max_num_objects
    for oid, (yc, xc) in enumerate(objects):
        dh, dw = size[0] // 2, size[1] // 2
        y0, x0, y1, x1 = np.array(
            [yc - dh, xc - dw, yc + dh, xc + dw]) + border // 2
        result[oid, :, :, :4] = padded_image[:, y0:y1, x0:x1, :]
        result[oid, :, :, 3] = gaussian_kernel(size[0], size[1], sigma, sigma)
    return np.split(result, [3], 3)


def detect_parts_heatmaps(heatmaps, centers, size, num_parts=14):
    """
    Given heat-maps find the position of each joint by means of n argmax
    function
    """
    parts = np.zeros((len(centers), num_parts, 2), dtype=np.int32)
    visible = np.zeros((len(centers), num_parts), dtype=bool)
    for oid, (yc, xc) in enumerate(centers):
        part_hmap = skimage.transform.resize(
            np.clip(heatmaps[oid], -1, 1), size)
        for pid in range(num_parts):
            y, x = np.unravel_index(np.argmax(part_hmap[:, :, pid]), size)
            parts[oid, pid] = y + yc - size[0] // 2, x + xc - size[1] // 2
            visible[oid, pid] = np.mean(
                part_hmap[:, :, pid]) > config.VISIBLE_PART
    return parts, visible


def import_json(path='json/MPI_annotations.json', order='json/MPI_order.npy'):
    """Get the json file containing the dataset.
    We want the data to be shuffled, however the training has to be repeatable.
    This means that once shuffled the order has to me mantained."""
    with open(path) as data_file:
        data_this = json.load(data_file)
        data_this = np.array(data_this['root'])
    num_samples = len(data_this)

    if os.path.exists(order):
        idx = np.load(order)
    else:
        idx = np.random.permutation(num_samples).tolist()
        np.save(order, idx)

    is_not_validation = [not data_this[i]['isValidation']
                         for i in range(num_samples)]
    keep_data_idx = list(compress(idx, is_not_validation))

    data = data_this[keep_data_idx]
    return data, len(keep_data_idx)


def generate_labels(image_shape, joint_positions, num_other_people,
                    joints_other_people, offset):
    """
    Given as input a set of joint positions and the size of the input image
    it generates
    a set of heat-maps of the same size. It generates both heat-maps used as
    labels for the first stage (label_1st_lower) and for all the other stages
    (label_lower).
    """
    _FILTER_JOINTS = np.array([9, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5])

    img_height, img_width, _ = image_shape
    heat_maps_single_p = np.zeros(
        (config.NUM_OUTPUT, config.INPUT_SIZE, config.INPUT_SIZE))
    heat_maps_other_p = np.zeros(
        (config.NUM_OUTPUT, config.INPUT_SIZE, config.INPUT_SIZE))

    # generate first set of heat-maps
    for i in range(config.NUM_JOINTS):
        # the set of joints can be different fromt the one in the json file
        curr_joint = joint_positions[_FILTER_JOINTS[i]]
        skip = (curr_joint[0] < 0 or curr_joint[1] < 0 or
                curr_joint[0] >= img_width or curr_joint[1] >= img_height)
        if not skip:
            heat_maps_single_p[i] = gaussian_heatmap(
                config.INPUT_SIZE, config.INPUT_SIZE,
                curr_joint[
                    1] - offset[1], curr_joint[0] - offset[0],
                config.SIGMA, config.SIGMA)

            heat_maps_other_p[i] = gaussian_heatmap(
                config.INPUT_SIZE, config.INPUT_SIZE,
                curr_joint[
                    1] - offset[1], curr_joint[0] - offset[0],
                config.SIGMA, config.SIGMA)

    heat_maps_single_p[-1] = np.maximum(
        1 - np.max(heat_maps_single_p[:-1], axis=0),
        np.zeros((config.INPUT_SIZE, config.INPUT_SIZE)))
    heat_maps_single_p = np.transpose(heat_maps_single_p, (1, 2, 0))

    # generate second set of heat-maps for other people in the image
    for p in range(int(num_other_people)):
        for i in range(config.NUM_JOINTS):
            # the set of joints can be different fromt the one in the json file
            try:
                if num_other_people == 1:
                    curr_joint = joints_other_people[_FILTER_JOINTS[i]]
                else:
                    curr_joint = joints_other_people[p][_FILTER_JOINTS[i]]
                skip = (
                    curr_joint[0] < 0 or curr_joint[1] < 0 or
                    curr_joint[0] >= img_width or curr_joint[1] >= img_height)
            except IndexError:
                skip = True
            if not skip:
                heat_maps_other_p[i] = gaussian_heatmap(
                    config.INPUT_SIZE, config.INPUT_SIZE,
                    curr_joint[1] - offset[1], curr_joint[0] - offset[0],
                    config.SIGMA, config.SIGMA, init=heat_maps_other_p[i])

    heat_maps_other_p[-1] = np.maximum(
        1 - np.max(heat_maps_other_p[:-1], axis=0),
        np.zeros((config.INPUT_SIZE, config.INPUT_SIZE)))

    heat_maps_other_p = np.transpose(heat_maps_other_p, (1, 2, 0))

    # rescaling heat-maps accoring to the right shape
    labels_single = rescale(heat_maps_single_p, config.OUTPUT_SIZE)
    labels_people = rescale(heat_maps_other_p, config.OUTPUT_SIZE)
    return labels_people, labels_single


def generate_center_map(center_pos, img_shape):
    """
    Given the position of the person and the size of the input image it
    generates
    a heat-map where a gaissian distribution is fit in the position of the
    person in the image.
    """
    img_height = img_shape
    img_width = img_shape
    center_map = gaussian_heatmap(
        img_height, img_width, center_pos[1], center_pos[0],
        config.SIGMA_CENTER, config.SIGMA_CENTER)
    return center_map


def rescale(data, new_size):
    """Rescale data to a fixed dimension, regardless the number of channels.
    Data has to be in the format (h,w,c)."""
    if data.ndim > 2:
        assert data.shape[2] < data.shape[0]
        assert data.shape[2] < data.shape[1]
    resized_data = cv2.resize(
        data, (new_size, new_size), interpolation=cv2.INTER_CUBIC)
    return resized_data


def crop_image(image, obj_pose):
    """
    Crop the image in order to have the person at the center and the final
    image size
    is the same as the expected CNN input size.
    Returns the cropped image and the offset that is used to update the joint
    positions.
    """
    offset_left = int(obj_pose[0] - config.INPUT_SIZE // 2)
    offset_up = int(obj_pose[1] - config.INPUT_SIZE // 2)
    # just for checking that it's inside the image
    offset_right = int(image.shape[1] - obj_pose[0] - config.INPUT_SIZE // 2)
    offset_bottom = int(image.shape[0] - obj_pose[1] - config.INPUT_SIZE // 2)

    pad_left, pad_right, pad_up, pad_bottom = 0, 0, 0, 0
    if offset_left < 0:
        pad_left = -offset_left
    if offset_right < 0:
        pad_right = -offset_right
    if offset_up < 0:
        pad_up = -offset_up
    if offset_bottom < 0:
        pad_bottom = -offset_bottom
    padded_image = np.lib.pad(
        image, ((pad_up, pad_bottom), (pad_left, pad_right), (0, 0)),
        'constant', constant_values=((0, 0), (0, 0), (0, 0)))

    cropped_image = padded_image[
        offset_up + pad_up: offset_up + pad_up + config.INPUT_SIZE,
        offset_left + pad_left: offset_left + pad_left + config.INPUT_SIZE]

    return cropped_image, np.array([offset_left, offset_up])
