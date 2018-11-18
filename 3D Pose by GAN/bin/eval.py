

import chainer
from chainer import cuda
from chainer import dataset
from chainer import iterators
from chainer import serializers

import chainer.functions as F

import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import projection_gan

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str,
                        help='Generatorの重みファイルへのパス')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--batchsize', '-b', type=int, default=200)
    parser.add_argument('--allow_inversion', action="store_true"
                         )
    args = parser.parse_args()

    with open(os.path.join(
            os.path.dirname(args.model_path), 'options.json')) as f:
        opts = json.load(f)

    gen = projection_gan.pose.posenet.MLP(mode='generator',
        use_bn=opts['use_bn'], activate_func=getattr(F, opts['activate_func']))
    serializers.load_npz(args.model_path, gen)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        gen.to_gpu()

    if opts['action'] == 'all':
        with open(os.path.join('data', 'actions.txt')) as f:
            actions = f.read().split('\n')[:-1]
    else:
        actions = [opts['action']]

    errors = []
    for act_name in actions:
        test = projection_gan.pose.dataset.pose_dataset.H36M(
            action=act_name, length=1, train=False,
            use_sh_detection=opts['use_sh_detection'])
        test_iter = iterators.MultiprocessIterator(
            test, args.batchsize, repeat=False, shuffle=False)
        eds = []
        for batch in test_iter:
            xy_proj, xyz, scale = dataset.concat_examples(
                batch, device=args.gpu)
            xy_proj, xyz = xy_proj[:, 0], xyz[:, 0]
            with chainer.no_backprop_mode(), \
                    chainer.using_config('train', False):
                xy_real = chainer.Variable(xy_proj)
                z_pred = gen(xy_real)

            lx = gen.xp.power(xyz[:, 0::3] - xy_proj[:, 0::2], 2)
            ly = gen.xp.power(xyz[:, 1::3] - xy_proj[:, 1::2], 2)
            lz = gen.xp.power(xyz[:, 2::3] - z_pred.data, 2)

            euclidean_distance = gen.xp.sqrt(lx + ly + lz).mean(axis=1)
            euclidean_distance *= scale[:, 0]
            euclidean_distance = gen.xp.mean(euclidean_distance)

            eds.append(euclidean_distance * len(batch))
        test_iter.finalize()
        print(act_name, sum(eds) / len(test))
        errors.append(sum(eds) / len(test))
    print('-' * 20)
    print('average', sum(errors) / len(errors))

    with open(args.model_path.replace('.npz', '.csv'), 'w') as f:
        for act_name, error in zip(actions, errors):
            f.write('{},{}\n'.format(act_name, error))
        f.write('{},{}\n'.format('average', sum(errors) / len(errors)))
