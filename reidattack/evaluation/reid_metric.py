import warnings

import numpy as np
import torch
import torch.nn.functional as F
from evaluation.reranking import re_ranking

from .utils import compute_distance_matrix


class ReidMetric:
    def __init__(
        self,
        max_rank: int = 20,
        distributed_rank: int = 0,
        metric: str = "euclidean",
        norm_feat: bool = True,
        square_dist: bool = True,
    ):

        self._max_rank = max_rank
        self._distributed_rank = distributed_rank
        self._metric = metric
        self._norm_feat = norm_feat
        self._square_dist = square_dist

        self._gallery_cached = False

        self.reset()

    @property
    def gallery_cached(self):
        return self._gallery_cached

    def reset(self, reset_all=True):
        self._q_feats = []
        self._q_pids = []
        self._q_camids = []

        if reset_all:
            self._g_feats = []
            self._g_pids = []
            self._g_camids = []

    @torch.inference_mode()
    def update(self, feat, pid, camid, is_query=True):
        if self._distributed_rank > 0:
            return

        feat, pid, camid = feat.cpu(), pid.cpu(), camid.cpu()
        if is_query:
            self._q_feats.append(feat)
            self._q_pids.extend(np.asarray(pid))
            self._q_camids.extend(np.asarray(camid))
        else:
            self._g_feats.append(feat)
            self._g_pids.extend(np.asarray(pid))
            self._g_camids.extend(np.asarray(camid))

    def compute(self, rerank=False, reset_all=True):

        if not reset_all:
            self._gallery_cached = True
        else:
            self._gallery_cached = False

        if self._distributed_rank > 0:
            return None

        # query
        q_feats = torch.cat(self._q_feats, dim=0)
        q_pids = np.asarray(self._q_pids)
        q_camids = np.asarray(self._q_camids)
        # gallery
        g_feats = torch.cat(self._g_feats, dim=0)
        g_pids = np.asarray(self._g_pids)
        g_camids = np.asarray(self._g_camids)

        if rerank:
            distmat = re_ranking(
                q_feats,
                g_feats,
                k1=self._max_rank,
                k2=int(0.3 * self._max_rank),
                lambda_value=0.3,
            )
            # from evaluation.re_ranking.query_expansion import aqe
            # q_feats, g_feats = aqe(q_feats, g_feats)
            # distmat = compute_distance_matrix(
            #     q_feats, g_feats, metric=self._metric,
            #     normalize=self._norm_feat, square=self._square_dist)
        else:
            distmat = compute_distance_matrix(
                q_feats,
                g_feats,
                metric=self._metric,
                normalize=self._norm_feat,
                square=self._square_dist,
            )

        cmc, mAP = eval_func(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank=self._max_rank
        )

        results = {"top1": cmc[0], "top5": cmc[4], "map": mAP}
        for name, result in results.items():
            results[name] = round(result, 3)

        return results


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=20):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape

    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)

    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.0  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP
