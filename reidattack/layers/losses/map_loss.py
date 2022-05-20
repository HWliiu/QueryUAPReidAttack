import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat


def euclidean_dist(x, y, square=False):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    if square:
        dist = dist.clamp(min=1e-12)
    else:
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


# def map_loss(atta_q_feat, g_feat, pids, bin_num=10):
#     assert atta_q_feat.shape == g_feat.shape
#     assert atta_q_feat.shape[0] == pids.shape[0]
#     N = atta_q_feat.size(0)

#     atta_q_feat = F.normalize(atta_q_feat, dim=1, p=2)
#     g_feat = F.normalize(g_feat, dim=1, p=2)
#     dist_mat = euclidean_dist(atta_q_feat, g_feat)

#     # Range of dist_mat values (maximum - minimum) is 2.
#     # dist_mat_ptp = 2.   # peak to peak
#     dist_mat_ptp = dist_mat.max().ceil()   # peak to peak
#     bin_len = dist_mat_ptp / (bin_num - 1)
#     is_pos = pids.expand(N, N).eq(pids.expand(N, N).t()).float()

#     total_true_indicator = torch.zeros(N, device=atta_q_feat.device)
#     total_all_indicator = torch.zeros(N, device=atta_q_feat.device)
#     AP = torch.zeros(N, device=atta_q_feat.device)

#     for i in range(bin_num):
#         # bm is the center of each bin
#         bm = i * bin_len
#         # indicator is the weight of the dist_mat values
#         indicator = (1 - torch.abs(dist_mat - bm) / bin_len).clamp(min=0)
#         true_indicator = is_pos * indicator
#         all_indicator = indicator
#         sum_true_indicator = torch.sum(true_indicator, 1)
#         sum_all_indicator = torch.sum(all_indicator, 1)
#         total_true_indicator = total_true_indicator + sum_true_indicator
#         total_all_indicator = total_all_indicator + sum_all_indicator
#         Pm = total_true_indicator / total_all_indicator.clamp(min=1e-12)
#         rm = sum_true_indicator / is_pos.sum(dim=-1)
#         ap_bin = Pm * rm
#         AP = AP + ap_bin
#     final_AP = torch.sum(AP) / N
#     return final_AP

def map_loss(atta_q_feat, g_feat, pids, bin_num=10):
    assert atta_q_feat.shape == g_feat.shape
    assert atta_q_feat.shape[0] == pids.shape[0]
    N = atta_q_feat.size(0)

    atta_q_feat = F.normalize(atta_q_feat, dim=1, p=2)
    g_feat = F.normalize(g_feat, dim=1, p=2)
    dist_mat = euclidean_dist(atta_q_feat, g_feat)

    # Range of dist_mat values.
    dist_mat_ptp = dist_mat.max().ceil()   # peak to peak
    bin_len = dist_mat_ptp / (bin_num - 1)
    is_pos = (repeat(pids, 'N->M N', M=N) ==
              repeat(pids, 'N->N M', M=N)).float()

    device = atta_q_feat.device
    total_true_indicator = torch.zeros(bin_num, N, device=device)
    total_all_indicator = torch.zeros(bin_num, N, device=device)
    AP = torch.zeros(N, device=device)

    # bm is the center of each bin
    bm = torch.arange(bin_num, device=device) * bin_len
    # indicator is the weight of the dist_mat values
    indicator = (1 - torch.abs(rearrange(dist_mat,
                 'M N->1 M N') - rearrange(bm, 'B->B 1 1')) / bin_len).clamp(min=0)
    true_indicator = is_pos * indicator
    all_indicator = indicator
    sum_true_indicator = torch.sum(true_indicator, 1)
    sum_all_indicator = torch.sum(all_indicator, 1)
    total_true_indicator = total_true_indicator + sum_true_indicator
    total_all_indicator = total_all_indicator + sum_all_indicator
    Pm = total_true_indicator / total_all_indicator.clamp(min=1e-12)
    rm = sum_true_indicator / is_pos.sum(dim=-1)
    ap_bin = Pm * rm
    AP = AP + ap_bin.sum(dim=0)

    final_AP = torch.sum(AP) / N
    return final_AP


class APLoss(nn.Module):
    def __init__(self, bin_num=10) -> None:
        super().__init__()
        self.bin_num = bin_num

    def forward(self, atta_q_feat, g_feat, pids):
        return map_loss(atta_q_feat, g_feat, pids, self.bin_num)


# class APLoss (nn.Module):
#     """ Differentiable AP loss, through quantization. From the paper:

#         Learning with Average Precision: Training Image Retrieval with a Listwise Loss
#         Jerome Revaud, Jon Almazan, Rafael Sampaio de Rezende, Cesar de Souza
#         https://arxiv.org/abs/1906.07589
#     """

#     def __init__(self, nq=10):
#         super(APLoss, self).__init__()
#         assert isinstance(nq, int) and 2 <= nq <= 100
#         self.nq = nq
#         gap = 2.
#         # assert gap > 0
#         # Initialize quantizer as non-trainable convolution
#         self.quantizer = nn.Conv1d(1, 2 * nq, kernel_size=1, bias=True)
#         self.quantizer.requires_grad_(False)
#         a = (nq - 1) / gap
#         # First half equal to lines passing to (min+x,1) and (min+x+1/a,0) with x = {nq-1..0}*gap/(nq-1)
#         self.quantizer.weight[:nq] = -a
#         self.quantizer.bias[:nq] = torch.arange(
#             nq, 0, -1) + a * min  # b = 1 + a*(min+x)
#         # First half equal to lines passing to (min+x,1) and (min+x-1/a,0) with x = {nq-1..0}*gap/(nq-1)
#         self.quantizer.weight[nq:] = a
#         self.quantizer.bias[nq:] = torch.arange(
#             2 - nq, 2, 1) - a * min  # b = 1 - a*(min+x)
#         # First and last one as a horizontal straight line
#         self.quantizer.weight[0] = self.quantizer.weight[-1] = 0
#         self.quantizer.bias[0] = self.quantizer.bias[-1] = 1

#     def forward(self, atta_q_feat, g_feat, pids, qw=None):
#         assert atta_q_feat.shape == g_feat.shape
#         assert atta_q_feat.shape[0] == pids.shape[0]

#         atta_q_feat = F.normalize(atta_q_feat, dim=1, p=2)
#         g_feat = F.normalize(g_feat, dim=1, p=2)
#         dist_mat = euclidean_dist(atta_q_feat, g_feat)
#         N, M = dist_mat.shape
#         is_pos = pids.expand(N, N).eq(pids.expand(N, N).t()).float()

#         # Quantize all predictions
#         q = self.quantizer(dist_mat.unsqueeze(1))
#         q = torch.min(q[:, :self.nq], q[:, self.nq:]).clamp(min=0)  # N x Q x M

#         nbs = q.sum(dim=-1)  # number of samples  N x Q = c
#         # number of correct samples = c+ N x Q
#         rec = (q * is_pos.view(N, 1, M).float()).sum(dim=-1)
#         prec = rec.cumsum(dim=-1) / (1e-16 + nbs.cumsum(dim=-1))  # precision
#         rec /= rec.sum(dim=-1).unsqueeze(1)  # norm in [0,1]

#         ap = (prec * rec).sum(dim=-1)  # per-image AP

#         final_AP = ap.mean()
#         return final_AP
