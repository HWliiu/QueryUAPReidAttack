# Code imported from https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/data/sampler.py
import copy
import numpy as np
import random
from collections import defaultdict

from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler

from . import SAMPLER_REGISTRY

SAMPLER_REGISTRY.register(RandomSampler)
SAMPLER_REGISTRY.register(SequentialSampler)


def synchronize_random_state():
    from accelerate.state import AcceleratorState
    # current only support NCCL
    if AcceleratorState().distributed_type == "MULTI_GPU":
        import torch.distributed as dist
        random_state = random.getstate()
        numpy_state = np.random.get_state()
        object_list = [random_state, numpy_state]
        dist.broadcast_object_list(object_list)
        random.setstate(object_list[0])
        np.random.set_state(object_list[1])


@SAMPLER_REGISTRY.register()
class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid, dsetid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        if batch_size < num_instances:
            raise ValueError(
                'batch_size={} must be no less '
                'than num_instances={}'.format(batch_size, num_instances)
            )

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, items in enumerate(data_source):
            pid = items[1]
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        assert len(self.pids) >= self.num_pids_per_batch

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        synchronize_random_state()
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True
                )
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


@SAMPLER_REGISTRY.register()
class RandomDomainSampler(Sampler):
    """
    Random domain sampler.

    We consider each camera as a visual domain.

    How does the sampling work:
    1. Randomly sample N cameras (based on the "camid" label).
    2. From each camera, randomly sample K images.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid, dsetid).
        batch_size (int): batch size.
        n_domain (int): number of cameras to sample in a batch.
    """

    def __init__(self, data_source, batch_size, n_domain):
        self.data_source = data_source

        # Keep track of image indices for each domain
        self.domain_dict = defaultdict(list)
        for i, items in enumerate(data_source):
            camid = items[2]
            self.domain_dict[camid].append(i)
        self.domains = list(self.domain_dict.keys())

        # Make sure each domain can be assigned an equal number of images
        if n_domain is None or n_domain <= 0:
            n_domain = len(self.domains)
        assert batch_size % n_domain == 0
        self.n_img_per_domain = batch_size // n_domain

        self.batch_size = batch_size
        self.n_domain = n_domain
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        synchronize_random_state()
        domain_dict = copy.deepcopy(self.domain_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            selected_domains = random.sample(self.domains, self.n_domain)

            for domain in selected_domains:
                idxs = domain_dict[domain]
                selected_idxs = random.sample(idxs, self.n_img_per_domain)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:
                    domain_dict[domain].remove(idx)

                remaining = len(domain_dict[domain])
                if remaining < self.n_img_per_domain:
                    stop_sampling = True

        return iter(final_idxs)

    def __len__(self):
        return self.length


@SAMPLER_REGISTRY.register()
class RandomDatasetSampler(Sampler):
    """Random dataset sampler.

    How does the sampling work:
    1. Randomly sample N datasets (based on the "dsetid" label).
    2. From each dataset, randomly sample K images.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid, dsetid).
        batch_size (int): batch size.
        n_dataset (int): number of datasets to sample in a batch.
    """

    def __init__(self, data_source, batch_size, n_dataset):
        self.data_source = data_source

        # Keep track of image indices for each dataset
        self.dataset_dict = defaultdict(list)
        for i, items in enumerate(data_source):
            dsetid = items[3]
            self.dataset_dict[dsetid].append(i)
        self.datasets = list(self.dataset_dict.keys())

        # Make sure each dataset can be assigned an equal number of images
        if n_dataset is None or n_dataset <= 0:
            n_dataset = len(self.datasets)
        assert batch_size % n_dataset == 0
        self.n_img_per_dset = batch_size // n_dataset

        self.batch_size = batch_size
        self.n_dataset = n_dataset
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        synchronize_random_state()
        dataset_dict = copy.deepcopy(self.dataset_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            selected_datasets = random.sample(self.datasets, self.n_dataset)

            for dset in selected_datasets:
                idxs = dataset_dict[dset]
                selected_idxs = random.sample(idxs, self.n_img_per_dset)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:
                    dataset_dict[dset].remove(idx)

                remaining = len(dataset_dict[dset])
                if remaining < self.n_img_per_dset:
                    stop_sampling = True

        return iter(final_idxs)

    def __len__(self):
        return self.length
