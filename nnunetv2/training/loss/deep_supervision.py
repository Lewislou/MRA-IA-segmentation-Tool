from torch import nn

class DeepSupervisionWrapper(nn.Module):
    def __init__(self, loss, weight_factors=None):
        super(DeepSupervisionWrapper, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, *args):
        for i in args:
            assert isinstance(i, (tuple, list)), "all args must be either tuple or list, got %s" % type(i)

        if self.weight_factors is None:
            weights = [1] * len(args[0])
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(*[j[0] for j in args])
        for i, inputs in enumerate(zip(*args)):
            if i == 0:
                continue
            l += weights[i] * self.loss(*inputs)
        return l
