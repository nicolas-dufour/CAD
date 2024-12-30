import glob
import json
import logging
import os
import random
from collections import OrderedDict
from multiprocessing import Value
from pathlib import Path

import braceexpand
import numpy as np
import pandas as pd
import torch
import webdataset as wds
from lightning_fabric.utilities.rank_zero import _get_rank
from PIL import Image
from torch.utils.data import Dataset, get_worker_info
from tqdm import tqdm
from webdataset.tariterators import (
    base_plus_ext,
    tar_file_expander,
    url_opener,
    valid_sample,
)

from cad.utils.one_hot_transform import OneHotTransform

# class LDMImageNet(Dataset):
#     def __init__(self, root, transforms=None, target_transform=None):
#         self.image_transforms = transforms
#         self.root = root
#         self.target_transform = target_transform
#         self.labels = os.listdir(root)
#         self.labels.sort()
#         self.labels = {label: i for i, label in enumerate(self.labels)}
#         self.paths = glob.glob(os.path.join(root, "*", "*.npy"))

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, idx):
#         path = self.paths[idx]
#         label = os.path.basename(os.path.dirname(path))
#         label = self.labels[label]
#         data = np.load(path)
#         data = torch.from_numpy(data).float()
#         if self.target_transform is not None:
#             label = self.target_transform(label)
#         return data, label


class LDMImageNet(Dataset):
    def __init__(
        self,
        root,
        image_transforms=None,
        vae_embedding_mean_name=None,
        vae_embedding_std_name=None,
        return_image=True,
        target_transform=None,
    ):
        self.root = Path(root)
        self.image_transforms = image_transforms
        self.metadata = pd.read_csv(
            root / Path("global_metadata.csv"), converters={"key": str}
        )
        self.vae_embedding_mean_name = vae_embedding_mean_name
        self.vae_embedding_std_name = vae_embedding_std_name
        self.return_image = return_image
        self.target_transform = target_transform

    def __getitem__(self, idx):
        return_dict = {}
        metadata = self.metadata.iloc[idx]
        key = metadata["key"]
        if self.return_image:
            image_path = self.root / "images" / f"{key}.jpg"
            image = Image.open(image_path).convert("RGB")
            if self.image_transforms is not None:
                image = self.image_transforms(image)
            return_dict["image"] = image
        if self.vae_embedding_mean_name is not None:
            vae_embedding_path = self.root / self.vae_embedding_mean_name / f"{key}.npy"
            vae_embedding = torch.from_numpy(np.load(vae_embedding_path))
            return_dict[self.vae_embedding_mean_name] = vae_embedding
            if self.vae_embedding_std_name is not None:
                vae_embedding_std_path = (
                    self.root / self.vae_embedding_std_name / f"{key}.npy"
                )
                vae_embedding_std = torch.from_numpy(np.load(vae_embedding_std_path))
                return_dict[self.vae_embedding_std_name] = vae_embedding_std
        return_dict["label"] = self.target_transform(int(metadata["label"]))
        return return_dict

    def __len__(self):
        return len(self.metadata)


class LDMImagenetWebdataset(wds.DataPipeline):
    def __init__(
        self,
        root,
        image_transforms=None,
        distributed=True,
        train=True,
        epoch=0,
        seed=3407,
        vae_embedding_name_mean=None,
        vae_embedding_name_std=None,
        return_image=True,
        num_classes=1000,
        shard_shuffle_size=2000,
        shard_shuffle_initial=500,
        sample_shuffle_size=5000,
        sample_shuffle_initial=1000,
    ):
        self.image_transforms = image_transforms
        dataset_tar_files = []
        # Get a list of all tar files in the directory
        if " " in root:
            root = root.split(" ")
            print(f"Using multiple dataset[s: {root}")
        if isinstance(root, str):
            tar_files = [f for f in os.listdir(root) if f.endswith(".tar")]

            # Sort the list of tar files
            tar_files.sort()

            first_tar_file = tar_files[0].split(".")[0]
            last_tar_file = tar_files[-1].split(".")[0]

            for tar_file in tar_files:
                dataset_tar_files.append(f"{root}/{tar_file}")

            dataset_pattern = f"{root}/{{{first_tar_file}..{last_tar_file}}}.tar"
            self.num_samples, _ = get_dataset_size(dataset_pattern)
        elif isinstance(root, list):
            num_samples = 0
            for r in root:
                tar_files = [f for f in os.listdir(r) if f.endswith(".tar")]
                tar_files.sort()
                first_tar_file = tar_files[0].split(".")[0]
                last_tar_file = tar_files[-1].split(".")[0]

                for tar_file in tar_files:
                    dataset_tar_files.append(f"{r}/{tar_file}")

                num_samples += get_dataset_size(
                    f"{r}/{{{first_tar_file}..{last_tar_file}}}.tar"
                )[0]
            self.num_samples = num_samples
        else:
            raise ValueError(
                f"root must be a string or list of strings. Got {type(root)}"
            )
        rank = _get_rank()
        self.shared_epoch = SharedEpoch(epoch)
        self.one_hot_transform = OneHotTransform(num_classes=num_classes)
        pipeline = [wds.SimpleShardList(dataset_tar_files)]

        if distributed:
            pipeline.extend(
                [
                    (
                        detshuffle2(
                            bufsize=shard_shuffle_size,
                            initial=shard_shuffle_initial,
                            seed=seed,
                            epoch=self.shared_epoch,
                        )
                        if train
                        else None
                    ),
                    wds.split_by_node,
                    wds.split_by_worker,
                    tarfile_to_samples_nothrow,
                    (
                        wds.shuffle(
                            bufsize=sample_shuffle_size,
                            initial=sample_shuffle_initial,
                        )
                        if train
                        else None
                    ),
                ]
            )
        else:
            pipeline.extend(
                [
                    (
                        wds.shuffle(
                            bufsize=shard_shuffle_size,
                            initial=sample_shuffle_initial,
                        )
                        if train
                        else None
                    ),
                    wds.split_by_worker,
                    tarfile_to_samples_nothrow,
                    (
                        wds.shuffle(
                            bufsize=sample_shuffle_size,
                            initial=sample_shuffle_initial,
                        )
                        if train
                        else None
                    ),
                ]
            )
        outputs_transforms = OrderedDict()
        outputs_rename = OrderedDict()
        if return_image:
            outputs_rename["image.jpg"] = "jpg;png;webp;jpeg"
            outputs_transforms["image.jpg"] = self.image_transforms
        if vae_embedding_name_mean is not None:
            outputs_rename[
                f"{vae_embedding_name_mean}.npy"
            ] = f"{vae_embedding_name_mean}.npy"
            outputs_transforms[
                f"{vae_embedding_name_mean}.npy"
            ] = lambda x: torch.from_numpy(x)
        if vae_embedding_name_std is not None:
            outputs_rename[
                f"{vae_embedding_name_std}.npy"
            ] = f"{vae_embedding_name_std}.npy"
            outputs_transforms[
                f"{vae_embedding_name_std}.npy"
            ] = lambda x: torch.from_numpy(x)
        outputs_rename["label.json"] = "label.json"
        outputs_transforms["label.json"] = lambda x: self.one_hot_transform(x["id"])
        pipeline.extend(
            [
                wds.rename(**outputs_rename),
                filter_dict_keys(*outputs_rename.keys(), handler=log_and_continue),
            ]
        )
        if return_image:
            pipeline.append(wds.decode("pilrgb", handler=log_and_continue))
        else:
            pipeline.append(wds.decode("pilrgb", handler=log_and_continue))
        pipeline.extend(
            [
                wds.map_dict(**outputs_transforms, handler=log_and_continue),
                wds.rename(
                    **{k.split(".")[0]: k for k in outputs_transforms.keys()},
                ),
            ]
        )

        super().__init__(*pipeline)

    def __len__(self):
        return self.num_samples


def get_dataset_size(shards):
    shards_list, _ = expand_urls(shards)
    dir_path = os.path.dirname(shards_list[0])
    sizes_filename = os.path.join(dir_path, "sizes.json")
    if os.path.exists(sizes_filename):
        sizes = json.load(open(sizes_filename, "r"))
        total_size = sum([int(sizes[os.path.basename(shard)]) for shard in shards_list])
    else:
        total_size = 0  # num samples undefined
        sizes = {}
        for shard in tqdm(shards_list):
            dataset = wds.WebDataset(shard)
            num_samples = sum(1 for _ in dataset)
            total_size += num_samples
            sizes[os.path.basename(shard)] = num_samples
        print(f"Total number of samples: {total_size}")
        with open(sizes_filename, "w") as f:
            json.dump(sizes, f)

    num_shards = len(shards_list)
    return total_size, num_shards


def expand_urls(urls, weights=None):
    if weights is None:
        expanded_urls = wds.shardlists.expand_urls(urls)
        return expanded_urls, None
    if isinstance(urls, str):
        urllist = urls.split("::")
        weights = weights.split("::")
        assert len(weights) == len(
            urllist
        ), f"Expected the number of data components ({len(urllist)}) and weights({len(weights)}) to match."
        weights = [float(weight) for weight in weights]
        all_urls, all_weights = [], []
        for url, weight in zip(urllist, weights):
            expanded_url = list(braceexpand.braceexpand(url))
            expanded_weights = [weight for _ in expanded_url]
            all_urls.extend(expanded_url)
            all_weights.extend(expanded_weights)
        return all_urls, all_weights
    else:
        all_urls = list(urls)
        return all_urls, weights


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value("i", epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


# _SHARD_SHUFFLE_SIZE = 256
# _SHARD_SHUFFLE_INITIAL = 128
# _SAMPLE_SHUFFLE_SIZE = 5000
# _SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
        self,
        bufsize=1000,
        initial=100,
        seed=0,
        epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            # If seed is negative, we use the worker's seed, this will be different across all nodes/workers
            seed = pytorch_worker_seed(epoch)
        else:
            # This seed to be deterministic AND the same across all nodes/workers in each epoch
            seed = self.seed + epoch
        rng.seed(seed)
        return wds.filters._shuffle(src, self.bufsize, self.initial, rng)


def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f"Handling webdataset error ({repr(exn)}). Ignoring.")
    return True


def group_by_keys_nothrow(
    data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None
):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if (
            current_sample is None
            or prefix != current_sample["__key__"]
            or suffix in current_sample
        ):
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


def filter_no_caption_or_no_image(sample):
    has_caption = "txt" in sample
    has_image = (
        "png" in sample or "jpg" in sample or "jpeg" in sample or "webp" in sample
    )
    return has_caption and has_image


def filter_metadata(sample, min_image_size, min_clip_score):
    metadata = json.loads(sample["json"])
    width = metadata["width"]
    height = metadata["height"]
    clip_score = metadata["clip_score"] / 100
    return (
        width >= min_image_size
        and height >= min_image_size
        and clip_score >= min_clip_score
    )


def _filter_dict_keys(
    data,
    *args,
    handler=wds.reraise_exception,
    missing_is_error=True,
    none_is_error=None,
):
    """Convert dict samples to tuples."""
    if none_is_error is None:
        none_is_error = missing_is_error
    if len(args) == 1 and isinstance(args[0], str) and " " in args[0]:
        args = args[0].split()

    for sample in data:
        try:
            result = {
                f: wds.getfirst(sample, f, missing_is_error=missing_is_error)
                for f in args
            }
            print
            if none_is_error and any(x is None for x in result):
                raise ValueError(f"to_tuple {args} got {sample.keys()}")
            yield result
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


filter_dict_keys = wds.pipelinefilter(_filter_dict_keys)