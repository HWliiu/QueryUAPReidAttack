from yacs.config import CfgNode as CN


# ----------------------------------------------------------------------------- ##
# Config definition
# ----------------------------------------------------------------------------- ##

_C = CN()


# ----------------------------------------------------------------------------- #
# DATA
# ----------------------------------------------------------------------------- #
_C.DATA = CN()

# -----------------------------------------------------------------------------
# TRAINSFORM
# -----------------------------------------------------------------------------
_C.DATA.TRANSFORM = CN()
# Size of the image during training
_C.DATA.TRANSFORM.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.DATA.TRANSFORM.SIZE_TEST = [256, 128]

# Normalize
_C.DATA.TRANSFORM.NORM = CN({"ENABLED": True})
# Values to be used for image normalization
_C.DATA.TRANSFORM.NORM.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.DATA.TRANSFORM.NORM.PIXEL_STD = [0.229, 0.224, 0.225]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATA.DATASET = CN()
# The dataset type
_C.DATA.DATASET.TYPE = "image"
# List of the dataset names for training
_C.DATA.DATASET.TRAIN_NAMES = ("Market1501",)
# The dataset name for testing
_C.DATA.DATASET.TEST_NAME = "Market1501"
# Combine trainset and testset joint training
_C.DATA.DATASET.COMBINEALL = False
# Root directory where datasets should be used
_C.DATA.DATASET.ROOT_DIR = "datasets"

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATA.DATALOADER = CN()
# Options: RandomIdentitySampler, RandomSampler, SequentialSampler
_C.DATA.DATALOADER.SAMPLER_TRAIN = "RandomIdentitySampler"
_C.DATA.DATALOADER.SAMPLE_NUM_TRAIN = 160
# Number of instance for each person
_C.DATA.DATALOADER.NUM_INSTANCE = 4
# Number of images per batch
_C.DATA.DATALOADER.BATCH_SIZE_TRAIN = 32
_C.DATA.DATALOADER.BATCH_SIZE_TEST = 128


# ----------------------------------------------------------------------------- #
# MODULE
# ----------------------------------------------------------------------------- #
_C.MODULE = CN()

# -----------------------------------------------------------------------------
# AGENT MODELS
# -----------------------------------------------------------------------------
_C.MODULE.AGENT_MODELS = CN()
# Name of the agent model
_C.MODULE.AGENT_MODELS.NAMES = ["resnet50_bot",]
# Agent model weight file path
_C.MODULE.AGENT_MODELS.WEIGHTS = [
    "model_weights/reid_models/ReidStrongBaseline/resnet50/market1501_resnet50_bot_map=0.86.pth",]

# -----------------------------------------------------------------------------
# SEGMENT MODEL
# -----------------------------------------------------------------------------
_C.MODULE.SEGMENT_MODEL = CN()
# Name of the segment model
_C.MODULE.SEGMENT_MODEL.NAME = "bisenetv2"
# Segment model weight file path
_C.MODULE.SEGMENT_MODEL.WEIGHT = "model_weights/segment_models/ppss_bisenetv2.pth"

# -----------------------------------------------------------------------------
# TARGET MODEL
# -----------------------------------------------------------------------------
_C.MODULE.TARGET_MODEL = CN()
# Name of the target model
_C.MODULE.TARGET_MODEL.NAME = "inceptionv3_bot"
# Target model weight file path
_C.MODULE.TARGET_MODEL.WEIGHT = "model_weights/reid_models/ReidStrongBaseline/inceptionv3/market1501_inceptionv3_bot_map=0.77.pth"

# ----------------------------------------------------------------------------- #
# TRAINER
# ----------------------------------------------------------------------------- #
_C.TRAINER = CN()
_C.TRAINER.MAX_EPOCH = 2
_C.TRAINER.EVAL_ONLY = False
_C.TRAINER.ATTACK_ALGORITHM = "query_uap"
_C.TRAINER.EPSILON = 10 / 255
# For test
# Whether to use reranking for testing
_C.TRAINER.RERANKING = False
_C.TRAINER.EVAL_PERIOD = 1
_C.TRAINER.UAP_WEIGHT = ""
