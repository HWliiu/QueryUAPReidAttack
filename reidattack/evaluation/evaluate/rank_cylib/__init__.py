# Code imported from https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/evaluation/rank_cylib/__init__.py
# from .roc_cy import evaluate_roc_cy
import ignite

from .rank_cy import evaluate_reid_cy


@ignite.distributed.utils.one_rank_only()
def compile_helper():
    """Compile helper function at runtime. Make sure this
    is invoked on a single process."""
    import os
    import subprocess

    path = os.path.abspath(os.path.dirname(__file__))
    ret = subprocess.run(["make", "-C", path])
    if ret.returncode != 0:
        print("Making cython reid evaluation module failed, exiting.")
        import sys

        sys.exit(1)
