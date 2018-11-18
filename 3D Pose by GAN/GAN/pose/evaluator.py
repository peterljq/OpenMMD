
import copy

import chainer
from chainer import function
import chainer.functions as F
from chainer import reporter as reporter_module
from chainer.training import extensions


class Evaluator(extensions.Evaluator):

    def evaluate(self):
        iterator = self._iterators['main']
        gen = self._targets['gen']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                xy_proj, xyz, scale = self.converter(batch, self.device)
                xy_proj, xyz = xy_proj[:, 0], xyz[:, 0]
                with function.no_backprop_mode(), \
                        chainer.using_config('train', False):
                    xy_real = chainer.Variable(xy_proj)
                    z_pred = gen(xy_real)
                    z_mse = F.mean_squared_error(z_pred, xyz[:, 2::3])
                    chainer.report({'z_mse': z_mse}, gen)

                    lx = gen.xp.power(xyz[:, 0::3] - xy_proj[:, 0::2], 2)
                    ly = gen.xp.power(xyz[:, 1::3] - xy_proj[:, 1::2], 2)
                    lz = gen.xp.power(xyz[:, 2::3] - z_pred.data, 2)

                    euclidean_distance = gen.xp.sqrt(lx + ly + lz).mean(axis=1)
                    euclidean_distance *= scale[:, 0]
                    euclidean_distance = gen.xp.mean(euclidean_distance)
                    chainer.report(
                        {'euclidean_distance': euclidean_distance}, gen)
            summary.add(observation)

        return summary.compute_mean()
