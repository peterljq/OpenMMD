

from __future__ import print_function
import argparse
import json
import multiprocessing
import time

import chainer
import chainer.functions as F
from chainer import training
from chainer.training import extensions

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from projection_gan.pose.posenet import MLP
from projection_gan.pose.dataset.pose_dataset import H36M, MPII
from projection_gan.pose.dataset.mpii_inf_3dhp_dataset import MPII3DDataset
from projection_gan.pose.updater import H36M_Updater
from projection_gan.pose.evaluator import Evaluator


def create_result_dir(dirname):
    if not os.path.exists('results'):
        os.mkdir('results')
    if dirname:
        result_dir = os.path.join('results', dirname)
    else:
        result_dir = os.path.join(
            'results', time.strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    return result_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-b', '--batchsize', type=int, default=16)
    parser.add_argument('-B', '--test_batchsize', type=int, default=32)
    parser.add_argument('-r', '--resume', default='')
    parser.add_argument('-o', '--out', type=str, default='')
    parser.add_argument('-e', '--epoch', type=int, default=20)
    parser.add_argument('-m', '--mode', type=str, default='unsupervised',
                        choices=['supervised', 'unsupervised'])
    parser.add_argument('-d', '--dataset', type=str, default='h36m',
                        choices=['h36m', 'mpii', 'mpi_inf'])
    parser.add_argument('-a', '--activate_func',
                        type=str, default='leaky_relu')
    parser.add_argument('-c', '--gan_accuracy_cap', type=float, default=0.9)
    parser.add_argument('-A', '--action', type=str, default='all')
    parser.add_argument('-s', '--snapshot_interval', type=int, default=1)
    parser.add_argument('-l', '--log_interval', type=int, default=1)
    parser.add_argument('--heuristic_loss_weight', type=float, default=1.0)
    parser.add_argument('--use_heuristic_loss', action="store_true")
    parser.add_argument('--use_sh_detection', action="store_true")
    parser.add_argument('--use_bn', action="store_true")
    args = parser.parse_args()
    args.out = create_result_dir(args.out)

    # Save options.
    with open(os.path.join(args.out, 'options.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(json.dumps(vars(args), indent=2))

    gen = MLP(mode='generator', use_bn=args.use_bn,
              activate_func=getattr(F, args.activate_func))
    dis = MLP(mode='discriminator', use_bn=args.use_bn,
              activate_func=getattr(F, args.activate_func))
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    # Setup an optimizer
    def make_optimizer(model):
        optimizer = chainer.optimizers.Adam(alpha=2e-4, beta1=0.5)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))
        return optimizer

    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)

    # Load dataset.
    if args.dataset == 'h36m':
        train = H36M(action=args.action, length=1, train=True,
                     use_sh_detection=args.use_sh_detection)
        test = H36M(action=args.action, length=1, train=False,
                    use_sh_detection=args.use_sh_detection)
    elif args.dataset == 'mpii':
        train = MPII(train=True, use_sh_detection=args.use_sh_detection)
        test = MPII(train=False, use_sh_detection=args.use_sh_detection)
    elif args.dataset == 'mpi_inf':
        train = MPII3DDataset(train=True)
        test = MPII3DDataset(train=False)
    print('TRAIN: {}, TEST: {}'.format(len(train), len(test)))

    multiprocessing.set_start_method('spawn')
    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)
    test_iter = chainer.iterators.MultiprocessIterator(
        test, args.test_batchsize, repeat=False, shuffle=False)

    # Set up a trainer
    updater = H36M_Updater(
        gan_accuracy_cap=args.gan_accuracy_cap,
        use_heuristic_loss=args.use_heuristic_loss,
        heuristic_loss_weight=args.heuristic_loss_weight,
        mode=args.mode, iterator={'main': train_iter, 'test': test_iter},
        optimizer={'gen': opt_gen, 'dis': opt_dis}, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    log_interval = (args.log_interval, 'epoch')
    snapshot_interval = (args.snapshot_interval, 'epoch')

    trainer.extend(Evaluator(test_iter, {'gen': gen}, device=args.gpu),
                   trigger=log_interval)
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch_{.updater.epoch}'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
        gen, 'gen_epoch_{.updater.epoch}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/z_mse', 'gen/loss', 'gen/loss_heuristic',
        'dis/loss', 'dis/acc', 'dis/acc/real', 'dis/acc/fake',
        'validation/gen/z_mse', 'validation/gen/euclidean_distance'
    ]), trigger=log_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
