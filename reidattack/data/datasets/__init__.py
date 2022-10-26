from utils.registry import Registry

IMAGE_DATASET_REGISTRY = Registry("IMAGE_DATASET")
IMAGE_DATASET_REGISTRY.__doc__ = """Registry for image datasets"""

VIDEO_DATASET_REGISTRY = Registry("VIDEO_DATASET")
VIDEO_DATASET_REGISTRY.__doc__ = """Registry for video datasets"""

from .bases import ReidDataset, ReidImageDataset, ReidVideoDataset
from .image import (CUHK01, CUHK02, CUHK03, CUHKSYSU, GRID, MSMT17, PRID,
                    DukeMTMCreID, Market1501, SenseReID, University1652, VIPeR,
                    iLIDS)
from .video import PRID2011, DukeMTMCVidReID, Mars, iLIDSVID
