import logging
import math
import random
from dataclasses import dataclass
from multiprocessing import Value
from ..imagenet_dataloader import build_common_augmentation
import webdataset as wds
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, tar_file_expander, valid_sample
import torch
import os
import re
try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def filter_no_caption_or_no_image(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    return has_caption and has_image


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True


def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.

    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        if 'fname' not in filesample.keys(): continue
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
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


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


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
        return _shuffle(src, self.bufsize, self.initial, rng)


def save_image(image):
    filename = f"results/sample_data/{random.randint(1, 1000000)}.jpg"
    image.save(filename)
    return image
    
def save_text(text):
    filename = f"{random.randint(1, 1000000)}.txt"
    with open(f"results/sample_data/{filename}", "w") as file:
        file.write(text)
    return text


def get_wds_dataset(args, world_size, is_train=True, epoch=0, floor=False):

    input_shards = args.data_path
    assert input_shards is not None

    num_samples = args.num_samples
    num_shards = args.num_shards

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    #--sample shards (for what?)
    pipeline = [wds.SimpleShardList(input_shards)]

    #data transform
    preprocess_img = build_common_augmentation(args.transforms)

    # at this point we have an iterator over all the shards
    if is_train:
        pipeline.extend([
            detshuffle2(
                bufsize=_SHARD_SHUFFLE_SIZE,
                initial=_SHARD_SHUFFLE_INITIAL,
                seed=0, #follow prototype.utils.torch_ddp_dist.set_random_seed
                epoch=shared_epoch,
            ),
            wds.split_by_node,
            wds.split_by_worker,
        ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        # wds.map_dict(image=save_image, text=save_text),
        wds.map_dict(image=preprocess_img, text=lambda text:text.strip()),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)


    if is_train:
        assert num_shards >= args.workers * world_size, 'number of shards must be >= total workers'

        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * world_size
        num_batches = round_fn(num_samples / global_batch_size) #total batches in a iteration
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers) #batch num for each wordk
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)


    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


import sys

from prototype.data.datasets.coco import COCOCaptionDataset

def get_coco_dataset(args, is_train, word_size, tokenizer=None):
    print("preparing COCO dataset")
    preprocess_fn = build_common_augmentation(args.transforms)
    dataset = COCOCaptionDataset(
        root_dir="/gscratch/krishna/datasets/coco/train2017",
        annotations_file="/gscratch/krishna/datasets/coco/annotations/captions_train2017.json",
        save_path="",
        transform=preprocess_fn,
        tokenzier=tokenizer,
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if word_size > 1 and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)




def sample_shard_paths(total_shards, sample_factor):
    """
    Randomly samples shard paths from a total number of shards, where the sample size is 1/sample_factor
    of the total.

    :param base_path: The base path containing the shard files
    :param total_shards: Total number of shard files
    :param sample_factor: The factor to determine sample size. Sample size will be total_shards/sample_factor
    :return: A list of sampled shard paths
    """
    base_path = '/data/yfcc-tmp/cc_12m/shards'
    # 1. Generate all the paths
    all_shards = [f"{base_path}/shard_0{i:05}.tar" for i in range(total_shards)]

    # 2. Determine the number of paths to sample
    num_to_sample = len(all_shards) // sample_factor

    # 3. Randomly sample paths
    sampled_shards = random.sample(all_shards, num_to_sample)

    return sampled_shards








import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from copy import deepcopy
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def swap_items(items):
    items = deepcopy(items)
    if len(items) > 1:
        swap_idxs = random.sample([x for x in range(len(items))], k=2)
        temp = items[swap_idxs[1]]
        items[swap_idxs[1]] = items[swap_idxs[0]]
        items[swap_idxs[0]] = temp
        
    return items


def swap_elements(caption):
    all_possible_negatives = []
    caption = caption.strip()
    # Tokenize and POS tag
    caption = caption.replace(" ,", ",").replace(" .", ".").replace(" :", ":").replace(" ;", ";").replace(" !", "!").replace(" ?", "?").replace(" -", "")  
    original_caption = deepcopy(caption)
    cleaned_caption = caption.replace(",", "").replace(".", "").replace(":", "").replace(";", "").replace("!", "").replace("?", "") 
    tokens = word_tokenize(cleaned_caption)
    tagged = pos_tag(tokens)

    for element_type in ["NN", "VB", "JJ", "RB", "VP", "NP", "ADJP"]:
        oris = nouns = [word for word, tag in tagged if tag.startswith(element_type)]
        swapped = swap_items(nouns)
        # Split the sentence into words while retaining special symbols
        words_with_symbols = re.findall(r'\b\w+\b|[.,:;!?]', caption)
        mapping_dict = {orig:new for orig, new in zip(oris, swapped)}
        # Replace words based on mapping
        modified_words = [mapping_dict[word] if word in mapping_dict else word for word in words_with_symbols]
        # Join words and symbols back into a sentence
        old_words = [word for word in words_with_symbols]
        old_caption = " ".join(old_words).replace(" ,", ",").replace(" .", ".").replace(" :", ":").replace(" ;", ";").replace(" !", "!").replace(" ?", "?").replace(" -", "-")  
        new_caption = " ".join(modified_words).replace(" ,", ",").replace(" .", ".").replace(" :", ":").replace(" ;", ";").replace(" !", "!").replace(" ?", "?").replace(" -", "-")  
        if new_caption != old_caption: all_possible_negatives.append(new_caption)
    if len(all_possible_negatives)>1: caption = random.choice(all_possible_negatives[1:])
    elif len(all_possible_negatives)>0: caption = all_possible_negatives[0]
    return [original_caption, caption]



def get_neg_wds_dataset(args, world_size, is_train=True, epoch=0, floor=False):

    input_shards = args.data_path
    assert input_shards is not None

    num_samples = args.num_samples
    num_shards = args.num_shards

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    #--sample shards (for what?)
    pipeline = [wds.SimpleShardList(input_shards)]

    #data transform
    preprocess_img = build_common_augmentation(args.transforms)

    # at this point we have an iterator over all the shards
    if is_train:
        pipeline.extend([
            detshuffle2(
                bufsize=_SHARD_SHUFFLE_SIZE,
                initial=_SHARD_SHUFFLE_INITIAL,
                seed=0, #follow prototype.utils.torch_ddp_dist.set_random_seed
                epoch=shared_epoch,
            ),
            wds.split_by_node,
            wds.split_by_worker,
        ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=swap_elements),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)


    if is_train:
        assert num_shards >= args.workers * world_size, 'number of shards must be >= total workers'

        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * world_size
        num_batches = round_fn(num_samples / global_batch_size) #total batches in a iteration
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers) #batch num for each wordk
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)


    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)






def get_unshuffled_wds_dataset(args, world_size, is_train=True, epoch=0, floor=False):
    input_shards = args.data_path
    assert input_shards is not None

    num_samples = args.num_samples
    num_shards = args.num_shards

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    pipeline = [wds.SimpleShardList(input_shards)]

    preprocess_img = build_common_augmentation(args.transforms)

    if is_train:
        pipeline.extend([
            wds.split_by_node,
            wds.split_by_worker,
            # Removed shuffling steps from here
            tarfile_to_samples_nothrow, 
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text:text.strip()),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        assert num_shards >= args.workers * world_size, 'number of shards must be >= total workers'

        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * world_size
        num_batches = round_fn(num_samples / global_batch_size) #total batches in a iteration
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers) #batch num for each wordk
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,  # Keep this as False
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # ... rest of the code remains unchanged ...
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)
