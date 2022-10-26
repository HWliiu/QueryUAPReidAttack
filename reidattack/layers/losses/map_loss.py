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
    dist_mat_ptp = dist_mat.max().ceil()  # peak to peak
    bin_len = dist_mat_ptp / (bin_num - 1)
    is_pos = (repeat(pids, "N->M N", M=N) == repeat(pids, "N->N M", M=N)).float()

    device = atta_q_feat.device
    total_true_indicator = torch.zeros(bin_num, N, device=device)
    total_all_indicator = torch.zeros(bin_num, N, device=device)
    AP = torch.zeros(N, device=device)

    # bm is the center of each bin
    bm = torch.arange(bin_num, device=device) * bin_len
    # indicator is the weight of the dist_mat values
    indicator = (
        1
        - torch.abs(rearrange(dist_mat, "M N->1 M N") - rearrange(bm, "B->B 1 1"))
        / bin_len
    ).clamp(min=0)
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
