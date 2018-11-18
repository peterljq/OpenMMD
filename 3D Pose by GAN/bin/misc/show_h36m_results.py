

import argparse
import chainer
import cv2
import json
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import evaluation_util
from projection_gan.pose.dataset.pose_dataset import H36M, MPII
from projection_gan.pose.dataset.mpii_inf_3dhp_dataset import MPII3DDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gen_path', type=str)
    parser.add_argument('--row', type=int, default=6)
    parser.add_argument('--col', type=int, default=6)
    parser.add_argument('--action', '-a', type=str, default='')
    parser.add_argument('--image', action='store_true')
    args = parser.parse_args()

    col, row = args.col, args.row
    gen_path = args.gen_path
    with open(os.path.join(
            os.path.dirname(gen_path), 'options.json'), 'rb') as f:
        opts = json.load(f)
    action = args.action if args.action else opts['action']

    imgs = np.zeros((350 * col, 600 * row, 3), dtype=np.uint8)
    gen = evaluation_util.load_model(opts)
    chainer.serializers.load_npz(gen_path, gen)

    if opts['dataset'] == 'h36m':
        test = H36M(action=opts['action'], length=1, train=False,
                    use_sh_detection=opts['use_sh_detection'])
    elif opts['dataset'] == 'mpii':
        test = MPII(train=False, use_sh_detection=opts['use_sh_detection'])
    elif opts['dataset'] == 'mpi_inf':
        test = MPII3DDataset(train=False)
    test_iter = chainer.iterators.SerialIterator(
        test, batch_size=row, shuffle=True, repeat=False)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        for k in range(col):
            batch = test_iter.next()
            batchsize = len(batch)
            xy_proj, xyz, scale = chainer.dataset.concat_examples(batch)
            xy_proj, xyz = xy_proj[:, 0], xyz[:, 0]
            xy_real = chainer.Variable(xy_proj)
            z_pred = gen(xy_real)

            # Rotation by 90 degree
            theta = np.array([np.pi / 2] * batchsize, dtype=np.float32)
            cos_theta = np.cos(theta)[:, None]
            sin_theta = np.sin(theta)[:, None]

            # 2D Projection
            # Ground Truth
            x = xyz[:, 0::3]
            y = xyz[:, 1::3]
            z = xyz[:, 2::3]
            new_x = x * cos_theta + z * sin_theta
            xy_gt = np.concatenate((new_x[:, :, None], y[:, :, None]), axis=2)
            xy_gt = np.reshape(xy_gt, (batchsize, -1))
            # Pediction
            x = xy_proj[:, 0::2]
            y = xy_proj[:, 1::2]
            new_x = x * cos_theta + z_pred.data * sin_theta
            xy_pred = np.concatenate((new_x[:, :, None], y[:, :, None]), axis=2)
            xy_pred = np.reshape(xy_pred, (batchsize, -1))

            for j in range(row):
                im0 = evaluation_util.create_img(xy_proj[j])
                im1 = evaluation_util.create_img(xy_gt[j])
                im2 = evaluation_util.create_img(xy_pred[j])
                imgs[k * 350:(k + 1) * 350, j * 600:(j + 1) * 600] = \
                    np.concatenate((im0, im1, im2), axis=1)

    for k in range(col + 1):
        cv2.line(imgs, (0, k * 350), (row * 600, k * 350), (0, 0, 255), 4)
    for j in range(row + 1):
        cv2.line(imgs, (j * 600, 0), (j * 600, col * 350), (0, 0, 255), 4)

    if not os.path.exists(os.path.join(os.path.dirname(gen_path), 'images')):
        os.mkdir(os.path.join(os.path.dirname(gen_path), 'images'))
    image_path = os.path.join(os.path.dirname(gen_path),
        'images', os.path.basename(gen_path).replace(
            '.npz', '_action_{}.png'.format(action)))
    cv2.imwrite(image_path, imgs)
    print('Saved image as \'{}\'.'.format(image_path))
