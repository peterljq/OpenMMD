

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np


class H36M_Updater(chainer.training.StandardUpdater):

    def __init__(self, gan_accuracy_cap, use_heuristic_loss,
                 heuristic_loss_weight, mode, *args, **kwargs):
        if not mode in ['supervised', 'unsupervised']:
            raise ValueError("only 'supervised' and 'unsupervised' are valid "
                             "for 'mode', but '{}' is given.".format(mode))
        self.gan_accuracy_cap = gan_accuracy_cap
        self.use_heuristic_loss = use_heuristic_loss
        self.heuristic_loss_weight = heuristic_loss_weight
        self.mode = mode
        super(H36M_Updater, self).__init__(*args, **kwargs)

    @staticmethod
    def calculate_rotation(xy_real, z_pred):
        xy_split = F.split_axis(xy_real, xy_real.data.shape[1], axis=1)
        z_split = F.split_axis(z_pred, z_pred.data.shape[1], axis=1)
        # Vector v0 (neck -> nose) on zx-plain. v0=(a0, b0).
        a0 = z_split[9] - z_split[8]
        b0 = xy_split[9 * 2] - xy_split[8 * 2]
        n0 = F.sqrt(a0 * a0 + b0 * b0)
        # Vector v1 (right shoulder -> left shoulder) on zx-plain. v1=(a1, b1).
        a1 = z_split[14] - z_split[11]
        b1 = xy_split[14 * 2] - xy_split[11 * 2]
        n1 = F.sqrt(a1 * a1 + b1 * b1)
        # Return sine value of the angle between v0 and v1.
        return (a0 * b1 - a1 * b0) / (n0 * n1)

    @staticmethod
    def calculate_heuristic_loss(xy_real, z_pred):
        return F.average(F.relu(
            -H36M_Updater.calculate_rotation(xy_real, z_pred)))

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        gen, dis = gen_optimizer.target, dis_optimizer.target

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        xy_proj, xyz, scale = self.converter(batch, self.device)
        xy_proj, xyz = xy_proj[:, 0], xyz[:, 0]

        xy_real = Variable(xy_proj)
        z_pred = gen(xy_real)
        z_mse = F.mean_squared_error(z_pred, xyz[:, 2::3])

        if self.mode == 'supervised':
            gen.cleargrads()
            z_mse.backward()
            gen_optimizer.update()
            chainer.report({'z_mse': z_mse}, gen)

        elif self.mode == 'unsupervised':
            # Random rotation.
            theta = gen.xp.random.uniform(0, 2 * np.pi, batchsize).astype('f')
            cos_theta = gen.xp.cos(theta)[:, None]
            sin_theta = gen.xp.sin(theta)[:, None]

            # 2D Projection.
            x = xy_real[:, 0::2]
            y = xy_real[:, 1::2]
            new_x = x * cos_theta + z_pred * sin_theta
            xy_fake = F.concat((new_x[:, :, None], y[:, :, None]), axis=2)
            xy_fake = F.reshape(xy_fake, (batchsize, -1))

            y_real = dis(xy_real)
            y_fake = dis(xy_fake)

            acc_dis_fake = F.binary_accuracy(
                y_fake, dis.xp.zeros(y_fake.data.shape, dtype=int))
            acc_dis_real = F.binary_accuracy(
                y_real, dis.xp.ones(y_real.data.shape, dtype=int))
            acc_dis = (acc_dis_fake + acc_dis_real) / 2

            loss_gen = F.sum(F.softplus(-y_fake)) / batchsize
            if self.use_heuristic_loss:
                loss_heuristic = self.calculate_heuristic_loss(
                    xy_real=xy_real, z_pred=z_pred)
                loss_gen += loss_heuristic * self.heuristic_loss_weight
                chainer.report({'loss_heuristic': loss_heuristic}, gen)
            gen.cleargrads()
            if acc_dis.data >= (1 - self.gan_accuracy_cap):
                loss_gen.backward()
                gen_optimizer.update()
            xy_fake.unchain_backward()

            loss_dis = F.sum(F.softplus(-y_real)) / batchsize
            loss_dis += F.sum(F.softplus(y_fake)) / batchsize
            dis.cleargrads()
            if acc_dis.data <= self.gan_accuracy_cap:
                loss_dis.backward()
                dis_optimizer.update()

            chainer.report({'loss': loss_gen, 'z_mse': z_mse}, gen)
            chainer.report({
                'loss': loss_dis, 'acc': acc_dis, 'acc/fake': acc_dis_fake,
                'acc/real': acc_dis_real}, dis)
