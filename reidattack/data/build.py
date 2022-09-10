from collections import defaultdict
import logging
import random

import torch
from torch.utils.data import DataLoader

from .datasets import IMAGE_DATASET_REGISTRY, VIDEO_DATASET_REGISTRY
from .samplers import SAMPLER_REGISTRY
from .transforms import build_transforms


def _random_choose(train_set, num_id, num_img):
    random.seed(1)
    index_dic = defaultdict(list)
    num_imgs_per_id = int(num_img / num_id)
    assert num_img % num_id == 0
    for i, (_, pid, _, _) in enumerate(train_set):
        index_dic[pid].append(i)
    pids = list(index_dic.keys())
    random.shuffle(pids)
    choose_pid = pids[:num_id]
    new_index_dic = defaultdict(list)
    for id in choose_pid:
        inds = index_dic[id]
        random.shuffle(inds)
        new_inds = inds[:num_imgs_per_id]
        new_index_dic[id] = new_inds
    new_train = []
    for key, value in new_index_dic.items():
        for ind in value:
            new_train.append(train_set[ind])
    return new_train


def build_train_dataloader(data_cfg, mode='train', **kwargs):
    """Build a dataloader for object re-identification with some default features.

    Returns:
        torch.utils.data.DataLoader: a dataloader.
    """
    logger = logging.getLogger(
        'reidattack.' + build_train_dataloader.__qualname__)

    # build dataset
    transforms = build_transforms(data_cfg.TRANSFORM, is_train=True)

    dataset_cfg = data_cfg.DATASET
    DATASET_REGISTRY = IMAGE_DATASET_REGISTRY \
        if dataset_cfg.TYPE == 'image' else VIDEO_DATASET_REGISTRY

    train_sets = list()
    for name in dataset_cfg.TRAIN_NAMES:
        train_set = DATASET_REGISTRY.get(name)(
            root=dataset_cfg.ROOT_DIR,
            mode=mode,
            transform=transforms,
            combineall=dataset_cfg.COMBINEALL)

        logger.info(train_set.show_train())
        train_sets.append(train_set)
    assert len(train_sets) > 0, 'No training set found'
    if len(train_sets) > 1:
        train_set = sum(train_sets)

    # build dataloader
    dataloader_cfg = data_cfg.DATALOADER
    batch_size = dataloader_cfg.BATCH_SIZE_TRAIN
    sampler_name = dataloader_cfg.SAMPLER_TRAIN
    train_sample_num = dataloader_cfg.SAMPLE_NUM_TRAIN

    if sampler_name == 'RandomIdentitySampler':
        num_instance = dataloader_cfg.NUM_INSTANCE
        assert batch_size > num_instance and batch_size % num_instance == 0, \
            'batch_size must be greater than num_instance and divisible by num_instance'

        train_set.data = _random_choose(
            train_set.data, train_sample_num // num_instance, train_sample_num)
        sampler = SAMPLER_REGISTRY.get(sampler_name)(
            data_source=train_set.data,
            batch_size=batch_size,
            num_instances=num_instance)
    elif sampler_name == 'RandomSampler':
        sampler = SAMPLER_REGISTRY.get(sampler_name)(
            data_source=train_set, num_samples=train_sample_num)
    elif sampler_name == 'SequentialSampler':
        sampler = SAMPLER_REGISTRY.get(sampler_name)(
            data_source=range(train_sample_num))
    else:
        # TODO: add other samplers
        raise NotImplementedError(sampler_name)

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=8,
        drop_last=False,
        pin_memory=True,
        **kwargs)

    return train_loader


def build_test_dataloader(data_cfg, **kwargs):
    """Similar to `build_train_dataloader`. This sampler coordinates all workers to produce
    the exact set of all samples
    """
    logger = logging.getLogger(
        'reidattack.' + build_test_dataloader.__qualname__)

    # build dataset
    transforms = build_transforms(data_cfg.TRANSFORM, is_train=False)

    dataset_cfg = data_cfg.DATASET
    DATASET_REGISTRY = IMAGE_DATASET_REGISTRY \
        if dataset_cfg.TYPE == 'image' else VIDEO_DATASET_REGISTRY

    dataset_name = dataset_cfg.TEST_NAME
    query_set = DATASET_REGISTRY.get(dataset_name)(
        root=dataset_cfg.ROOT_DIR,
        mode='query',
        transform=transforms)
    gallery_set = DATASET_REGISTRY.get(dataset_name)(
        root=dataset_cfg.ROOT_DIR,
        mode='gallery',
        transform=transforms)

    logger.info(query_set.show_test())

    # build dataloader
    dataloader_cfg = data_cfg.DATALOADER

    def build_test_loader(test_set):
        return DataLoader(
            dataset=test_set,
            batch_size=dataloader_cfg.BATCH_SIZE_TEST,
            num_workers=8,
            shuffle=False,
            drop_last=False,
            pin_memory=True)

    query_loader = build_test_loader(query_set)
    gallery_loader = build_test_loader(gallery_set)

    return query_loader, gallery_loader
