

import chainer
import chainer.functions as F
import chainer.links as L


class MLP(chainer.Chain):

    def __init__(self, n_in=34, n_unit=1024, mode='generator',
                 use_bn=False, activate_func=F.leaky_relu):
        if n_in % 2 != 0:
            raise ValueError("'n_in' must be divisible by 2.")
        if not mode in ['generator', 'discriminator']:
            raise ValueError("only 'generator' and 'discriminator' are valid "
                             "for 'mode', but '{}' is given.".format(mode))
        super(MLP, self).__init__()
        n_out = n_in // 2 if mode == 'generator' else 1
        print('MODEL: {}, N_OUT: {}, N_UNIT: {}'.format(mode, n_out, n_unit))
        self.mode = mode
        self.use_bn = use_bn
        self.activate_func = activate_func
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.l1 = L.Linear(n_in, n_unit, initialW=w)
            self.l2 = L.Linear(n_unit, n_unit, initialW=w)
            self.l3 = L.Linear(n_unit, n_unit, initialW=w)
            self.l4 = L.Linear(n_unit, n_out, initialW=w)

            if self.use_bn:
                self.bn1 = L.BatchNormalization(n_unit)
                self.bn2 = L.BatchNormalization(n_unit)
                self.bn3 = L.BatchNormalization(n_unit)

    def __call__(self, x):
        if self.use_bn:
            h1 = self.activate_func(self.bn1(self.l1(x)))
            h2 = self.activate_func(self.bn2(self.l2(h1)))
            h3 = self.activate_func(self.bn3(self.l3(h2)) + h1)
        else:
            h1 = self.activate_func(self.l1(x))
            h2 = self.activate_func(self.l2(h1))
            h3 = self.activate_func(self.l3(h2) + h1)
        return self.l4(h3)
