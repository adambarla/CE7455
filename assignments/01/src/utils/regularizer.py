import torch


class L1:
    def __init__(self, alpha, **kwargs):
        """
        :param alpha: weight of the regularizer
        """
        self.alpha = alpha

    def __call__(self, model):
        return self.alpha * sum(torch.abs(p).sum() for p in model.parameters())


class L2:
    def __init__(self, alpha, **kwargs):
        """
        :param alpha: weight of the regularizer
        """
        self.alpha = alpha

    def __call__(self, model):
        return self.alpha * sum((p**2).sum() for p in model.parameters())
