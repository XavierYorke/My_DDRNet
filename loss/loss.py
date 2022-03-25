import torch.nn as nn
import torch


class Tverskyloss(nn.Module):
    def __init__(self):
        super(Tverskyloss, self).__init__()

    def forward(self, outputs, target):
        batch_size = target.size(0)
        loss = 0.0
        beta = 0.7
        for i in range(batch_size):
            prob = outputs[0][i]
            ref = target[i]

            alpha = 1.0 - beta
            # TP ：ref * prob 两边都是positive
            # FP  ：(1 - ref) * prob 负的标签 正的预测
            # TN ：两边都是负的
            # FN ：ref*(1-prob）预测是负的
            tp = (ref * prob).sum()  # 真阳
            fp = ((1 - ref) * prob).sum()  # 假阳
            fn = (ref * (1 - prob)).sum()  # 假阴
            tversky = tp / (tp + alpha * fp + beta * fn)  # alpha beta 分别控制FP 和 FN的惩罚度
            loss = loss + (1 - tversky)
        return loss / batch_size


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)