"""
    Understanding Image Retrieval Re-Ranking: A Graph Neural Network Perspective

    Xuanmeng Zhang, Minyue Jiang, Zhedong Zheng, Xiao Tan, Errui Ding, Yi Yang

    Project Page : https://github.com/Xuanmeng-Zhang/gnn-re-ranking

    Paper: https://arxiv.org/abs/2012.07620v2

    ======================================================================
   
    On the Market-1501 dataset, we accelerate the re-ranking processing from 89.2s to 9.4ms
    with one K40m GPU, facilitating the real-time post-processing. Similarly, we observe 
    that our method achieves comparable or even better retrieval results on the other four 
    image retrieval benchmarks, i.e., VeRi-776, Oxford-5k, Paris-6k and University-1652, 
    with limited time cost.
"""

import torch
import torch.nn as nn
from setuptools import Extension, setup
from torch.autograd import Function
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="gnn_propagate",
    ext_modules=[
        CUDAExtension(
            "gnn_propagate",
            [
                "gnn_propagate.cpp",
                "gnn_propagate_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
