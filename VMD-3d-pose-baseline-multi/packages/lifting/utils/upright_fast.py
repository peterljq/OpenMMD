# -*- coding: utf-8 -*-
"""
Created on May 22 17:10 2017

@author: Denis Tome'
"""
import numpy as np
import scipy

__all__ = [
    'upgrade_r',
    'update_cam',
    'estimate_a_and_r_with_res',
    'estimate_a_and_r_with_res_weights',
    'pick_e'
]


def upgrade_r(r):
    """Upgrades complex parameterisation of planar rotation to tensor containing
    per frame 3x3 rotation matrices"""
    newr = np.zeros((3, 3))
    newr[:2, 0] = r
    newr[2, 2] = 1
    newr[1::-1, 1] = r
    newr[0, 1] *= -1
    return newr


def update_cam(cam):
    new_cam = cam[[0, 2, 1]].copy()
    new_cam = new_cam[:, [0, 2, 1]]
    return new_cam


def estimate_a_and_r_with_res(
        w, e, s0, camera_r, Lambda, check, a, weights, res, proj_e,
        residue, Ps, depth_reg, scale_prior):
    """
    TODO: Missing the following parameters in docstring:
        - w, e, s0, camera_r, Lambda, check, a, res, proj_e, depth_reg,
          scale_prior

    TODO: The following parameters are not used:
        - s0, weights

    So local optima are a problem in general.
    However:

        1. This problem is convex in a but not in r, and

        2. each frame can be solved independently.

    So for each frame, we can do a grid search in r and take the globally
    optimal solution.

    In practice, we just brute force over 100 different estimates of r, and
    take the best pair (r,a*(r)) where a*(r) is the optimal minimiser of a
    given r.

    Arguments:

        w is a 3d measurement matrix of form frames*2*points

        e is a 3d set of basis vectors of from basis*3*points

        s0 is the 3d rest shape of form 3*points

        Lambda are the regularisor coefficients on the coefficients of the
        weights typically generated using PPCA

        interval is how far round the circle we should check for break points
        we check every interval*2*pi radians

    Returns:

        a (basis coefficients) and r (representation of rotations as a complex
        number)
    """
    frames = w.shape[0]
    points = w.shape[2]
    basis = e.shape[0]
    r = np.empty(2)
    Ps_reshape = Ps.reshape(2 * points)
    w_reshape = w.reshape((frames, points * 2))

    for i in range(check.size):
        c = check[i]
        r[0] = np.cos(c)
        r[1] = np.sin(c)
        grot = camera_r.dot(upgrade_r(r))
        rot = grot[:2]
        res[:, :points * 2] = w_reshape
        res[:, :points * 2] -= Ps_reshape
        proj_e[:, :2 * points] = rot.dot(e).transpose(1, 0, 2).reshape(
            e.shape[0], 2 * points)

        if Lambda.size != 0:
            proj_e[:, 2 * points:2 * points + basis] = np.diag(Lambda[:Lambda.shape[0] - 1])
            res[:, 2 * points:].fill(0)
            res[:, :points * 2] *= Lambda[Lambda.shape[0] - 1]
            proj_e[:, :points * 2] *= Lambda[Lambda.shape[0] - 1]
            # depth regularizer not used
            proj_e[:, 2 * points + basis:] = ((Lambda[Lambda.shape[0] - 1] *
                                               depth_reg) * grot[2]).dot(e)
            # we let the person change scale
            res[:, 2 * points] = scale_prior

        """
        TODO: PLEASE REVIEW THE FOLLOWING CODE....
        overwrite_a and overwrite_b ARE UNEXPECTED ARGUMENTS OF
        scipy.linalg.lstsq
        """
        a[i], residue[i], _, _ = scipy.linalg.lstsq(
            proj_e.T, res.T, overwrite_a=True, overwrite_b=True)

    # find and return best coresponding solution
    best = np.argmin(residue, 0)
    assert (best.shape[0] == frames)
    theta = check[best]
    index = (best, np.arange(frames))
    aa = a.transpose(0, 2, 1)[index]
    retres = residue[index]
    r = np.empty((2, frames))
    r[0] = np.sin(theta)
    r[1] = np.cos(theta)
    return aa, r, retres


def estimate_a_and_r_with_res_weights(
        w, e, s0, camera_r, Lambda, check, a, weights, res, proj_e,
        residue, Ps, depth_reg, scale_prior):
    """
    TODO: Missing the following parameters in docstring:
     - w, e, s0, camera)r, Lambda, check, a, res, proj_e, residue,
     Ps, depth_reg, scale_prior

    So local optima are a problem in general.
    However:

        1. This problem is convex in a but not in r, and

        2. each frame can be solved independently.

    So for each frame, we can do a grid search in r and take the globally
    optimal solution.

    In practice, we just brute force over 100 different estimates of r, and
    take
    the best pair (r,a*(r)) where a*(r) is the optimal minimiser of a given r.

    Arguments:

        w is a 3d measurement matrix of form frames*2*points

        e is a 3d set of basis vectors of from basis*3*points

        s0 is the 3d rest shape of form 3*points

        Lambda are the regularisor coefficients on the coefficients of the
        weights
        typically generated using PPCA

        interval is how far round the circle we should check for break points
        we check every interval*2*pi radians

    Returns:

        a (basis coefficients) and r (representation of rotations as a complex
        number)
    """
    frames = w.shape[0]
    points = w.shape[2]
    basis = e.shape[0]
    r = np.empty(2)
    Ps_reshape = Ps.reshape(2 * points)
    w_reshape = w.reshape((frames, points * 2))
    p_copy = np.empty_like(proj_e)

    for i in range(check.size):
        c = check[i]
        r[0] = np.sin(c)
        r[1] = np.cos(c)
        grot = camera_r.dot(upgrade_r(r).T)
        rot = grot[:2]
        rot.dot(s0, Ps)  # TODO: remove?
        res[:, :points * 2] = w_reshape
        res[:, :points * 2] -= Ps_reshape
        proj_e[:, :2 * points] = rot.dot(e).transpose(1, 0, 2).reshape(
            e.shape[0], 2 * points)

        if Lambda.size != 0:
            proj_e[:, 2 * points:2 * points + basis] = np.diag(Lambda[:Lambda.shape[0] - 1])
            res[:, 2 * points:].fill(0)
            res[:, :points * 2] *= Lambda[Lambda.shape[0] - 1]
            proj_e[:, :points * 2] *= Lambda[Lambda.shape[0] - 1]
            proj_e[:, 2 * points + basis:] = ((Lambda[Lambda.shape[0] - 1] *
                                               depth_reg) * grot[2]).dot(e)
            res[:, 2 * points] = scale_prior
        if weights.size != 0:
            res[:, :points * 2] *= weights
        for j in range(frames):
            p_copy[:] = proj_e
            p_copy[:, :points * 2] *= weights[j]
            a[i, :, j], comp_residual, _, _ = np.linalg.lstsq(
                p_copy.T, res[j].T)
            if not comp_residual:
                # equations are over-determined
                residue[i, j] = 1e-5
            else:
                residue[i, j] = comp_residual
    # find and return best coresponding solution
    best = np.argmin(residue, 0)
    index = (best, np.arange(frames))
    theta = check[best]
    aa = a.transpose(0, 2, 1)[index]
    retres = residue[index]
    r = np.empty((2, frames))
    r[0] = np.sin(theta)
    r[1] = np.cos(theta)
    return aa, r, retres


def pick_e(w, e, s0, camera_r=None, Lambda=None,
           weights=None, scale_prior=-0.0014, interval=0.01, depth_reg=0.0325):
    """Brute force over charts from the manifold to find the best one.
        Returns best chart index and its a and r coefficients
        Returns assignment, and a and r coefficents"""

    camera_r = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]]
                          ) if camera_r is None else camera_r
    Lambda = np.ones((0, 0)) if Lambda is None else Lambda
    weights = np.ones((0, 0, 0)) if weights is None else weights

    charts = e.shape[0]
    frames = w.shape[0]
    basis = e.shape[1]
    points = e.shape[3]
    assert (s0.shape[0] == charts)
    r = np.empty((charts, 2, frames))
    a = np.empty((charts, frames, e.shape[1]))
    score = np.empty((charts, frames))
    check = np.arange(0, 1, interval) * 2 * np.pi
    cache_a = np.empty((check.size, basis, frames))
    residue = np.empty((check.size, frames))

    if Lambda.size != 0:
        res = np.zeros((frames, points * 2 + basis + points))
        proj_e = np.zeros((basis, 2 * points + basis + points))
    else:
        res = np.empty((frames, points * 2))
        proj_e = np.empty((basis, 2 * points))
    Ps = np.empty((2, points))

    if weights.size == 0:
        for i in range(charts):
            if Lambda.size != 0:
                a[i], r[i], score[i] = estimate_a_and_r_with_res(
                    w, e[i], s0[i], camera_r,
                    Lambda[i], check, cache_a, weights,
                    res, proj_e, residue, Ps,
                    depth_reg, scale_prior)
            else:
                a[i], r[i], score[i] = estimate_a_and_r_with_res(
                    w, e[i], s0[i], camera_r, Lambda,
                    check, cache_a, weights,
                    res, proj_e, residue, Ps,
                    depth_reg, scale_prior)
    else:
        w2 = weights.reshape(weights.shape[0], -1)
        for i in range(charts):
            if Lambda.size != 0:
                a[i], r[i], score[i] = estimate_a_and_r_with_res_weights(
                    w, e[i], s0[i], camera_r,
                    Lambda[i], check, cache_a, w2,
                    res, proj_e, residue, Ps,
                    depth_reg, scale_prior)
            else:
                a[i], r[i], score[i] = estimate_a_and_r_with_res_weights(
                    w, e[i], s0[i], camera_r, Lambda,
                    check, cache_a, w2,
                    res, proj_e, residue, Ps,
                    depth_reg, scale_prior)

    remaining_dims = 3 * w.shape[2] - e.shape[1]
    assert (np.all(score > 0))
    assert (remaining_dims >= 0)
    # Zero problems in log space due to un-regularised first co-efficient
    l = Lambda.copy()
    l[Lambda == 0] = 1
    llambda = -np.log(l)
    score /= 2
    return score, a, r
