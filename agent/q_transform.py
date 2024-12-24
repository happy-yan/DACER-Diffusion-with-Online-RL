import torch
from torch import nn

class QTransform(object):
    def __call__(self, q:torch.Tensor, **kwargs):
        raise NotImplementedError

class QReLU(QTransform):
    def __call__(self, q, **kwargs):
        q_neg = kwargs.get("q_neg", 0.0)
        q = q.clamp_(min=-q_neg).add_(q_neg)
        return q

class QEXPN(QTransform):
    def __call__(self, q, **kwargs):
        running_q_std = kwargs.get("running_q_std", None)
        running_q_mean = kwargs.get("running_q_mean", None)
        beta = kwargs.get("beta", 1.0)
        q = (q - running_q_mean) / running_q_std
        q.clamp_(min=-8.0, max=8.0)
        q = torch.exp(beta * q)
        return q

class QCut(QTransform):
    def __call__(self, q, **kwargs):
        cut = kwargs.get("cut", 1.0)
        q[q<cut] = cut
        return q

class QCutN(QTransform):

    def __init__(self, cut=0.0):
        self.cut = cut

    def __call__(self, q, **kwargs):
        running_q_mean = kwargs.get("running_q_mean", None)
        running_q_std = kwargs.get("running_q_std", None)
        beta = kwargs.get("beta", 1.0)
        q = beta * (q - running_q_mean)# / running_q_std
        q[q<self.cut] = self.cut
        return q

class QAdv(QTransform):
    def __call__(self, q, **kwargs):
        v = kwargs.get("v", None)
        chosen = kwargs.get("chosen", None)
        batch_size = kwargs.get("batch_size", None)
        adv = q.view(batch_size, chosen, 1) - v
        adv = adv.clamp_(min=0.0)
        return adv.view(batch_size * chosen, 1)


qrelu = QReLU()
qcut = QCut()
qexpn = QEXPN()
qcut0n = QCutN(0.0)
qcut1n = QCutN(1.0)
qadv = QAdv()
