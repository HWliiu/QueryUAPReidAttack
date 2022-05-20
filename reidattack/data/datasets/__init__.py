from utils.registry import Registry
IMAGE_DATASET_REGISTRY = Registry("IMAGE_DATASET")
IMAGE_DATASET_REGISTRY.__doc__ = """Registry for image datasets"""

VIDEO_DATASET_REGISTRY = Registry("VIDEO_DATASET")
VIDEO_DATASET_REGISTRY.__doc__ = """Registry for video datasets"""

from .image import (
    GRID, PRID, CUHK01, CUHK02, CUHK03, MSMT17, CUHKSYSU, VIPeR, SenseReID,
    Market1501, DukeMTMCreID, University1652, iLIDS
)
from .video import PRID2011, Mars, DukeMTMCVidReID, iLIDSVID
from .bases import ReidDataset, ReidImageDataset, ReidVideoDataset
