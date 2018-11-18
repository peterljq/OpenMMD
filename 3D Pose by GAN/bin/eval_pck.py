import argparse
import pickle
import sys
import tqdm
import os

import chainer

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
import evaluation_util

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import projection_gan

PCK_THRESHOLD = 150  # see "Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision"


def main(args):
    model_path = args.model_path
    with open(os.path.join(os.path.dirname(model_path), 'options.pickle'), 'rb') as f:
        options = pickle.load(f)
        model = projection_gan.pose.posenet.Linear(
            l_latent=options.l_latent, l_seq=options.l_seq, mode='generator',
            bn=options.bn, activate_func=getattr(chainer.functions, options.act_func))
    chainer.serializers.load_npz(model_path, model)
    if args.use_mpii_inf_3dhp:
        train = projection_gan.pose.dataset.mpii_inf_3dhp_dataset.MPII3DDataset(
            annotations_glob="/mnt/dataset/MPII_INF_3DHP/mpi_inf_3dhp/*/*/annot.mat", train=True)
    train_iter = chainer.iterators.SerialIterator(train, batch_size=args.batch, shuffle=True, repeat=False)

    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        total = 0
        true_positive = 0
        for i, batch in enumerate(tqdm.tqdm(train_iter, total=(len(train) / args.batch))):
            batch = chainer.dataset.concat_examples(batch)
            xy, xyz, scale = batch
            xy_real = chainer.Variable(xy)

            z_pred = model(xy_real).data[:, :, :, evaluation_util.JointsForPCK.from_h36m_joints]
            z_real = xyz[:, :, :, 2::3][:, :, :, evaluation_util.JointsForPCK.from_h36m_joints]

            per_joint_error = model.xp.sqrt((z_pred - z_real) * (z_pred - z_real)) * scale.reshape((-1, 1, 1, 1))
            true_positive += (per_joint_error < PCK_THRESHOLD).sum()
            total += per_joint_error.size
            if i % 1000 == 0:
                print(float(true_positive) / total * 100)
        print(float(true_positive) / total * 100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--action', '-a', type=str, default='')
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--use_mpii', action="store_true")
    parser.add_argument('--use_mpii_inf_3dhp', action="store_true")
    args = parser.parse_args()

    main(args)
