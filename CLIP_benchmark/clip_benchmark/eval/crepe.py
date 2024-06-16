import os
import json
import logging
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataclasses import dataclass
from clip_benchmark.models import MODEL_TYPES, load_clip
from crepe_eval_utils import BaseCsvDataset, get_one2many_metrics, get_one2many_rank, get_metrics
from crepe_params import setup_args
import torchvision
from torchvision import transforms as TT
from tqdm import tqdm

class DefaultArgs:
    def __init__(self):
        self.compo_type = "productivity"
        self.input_dir = "datasets/prod_hard_negatives/"
        # self.splits = list(range(4, 13))
        self.splits = [5, 10]
        self.hard_neg_types = ['atom', 'swap', 'negate']
        self.csv_img_key = "image_id"
        self.csv_caption_key = 'caption'
        self.crop = True
        self.hard_neg_key = 'hard_negs'
        self.one2many = True
    
class SystemArgs:
    def __init__(self):
        self.compo_type = "systematicity"
        self.train_dataset = 'cc12m'
        self.splits = ['seen_compounds', 'unseen_compounds']
        self.input_dir = "datasets/syst_hard_negatives/"
        self.hard_neg_types = ['atom', 'comp']
        self.csv_img_key = "image_id"
        self.csv_caption_key = 'caption'
        self.crop = True
        self.hard_neg_key = 'hard_negs'
        self.one2many = True
    
    
DATA2MODEL = {
    'cc12m': {
        'RN50-quickgelu': 'rn50-quickgelu-cc12m-f000538c.pt'
    },
    'yfcc': {
        'RN50-quickgelu': 'rn50-quickgelu-yfcc15m-455df137.pt', 
        'RN101-quickgelu': 'rn101-quickgelu-yfcc15m-3e04b30e.pt'
    },
    'laion': {
        'ViT-B-16':'vit_b_16-laion400m_e32-55e67d44.pt',
        'ViT-B-16-plus-240': 'vit_b_16_plus_240-laion400m_e32-699c4b84.pt',
        'ViT-B-32-quickgelu': 'vit_b_32-quickgelu-laion400m_e32-46683a32.pt',
        'ViT-L-14': 'vit_l_14-laion400m_e32-3d133497.pt',
    }
}

COMPO_SPLITS = ['seen_compounds', 'unseen_compounds']
COMPLEXITIES = list(range(4, 13))

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler

class CsvDataset(BaseCsvDataset):
    def __init__(self, input_filename, args, transforms):
        super().__init__(input_filename, args, transforms=transforms)

    def __getitem__(self, idx):
        raw_image = self.get_image_by_id(self.images[idx])
        if self.crop:
            raw_image = TF.crop(raw_image, self.ys[idx], self.xs[idx], self.heights[idx], self.widths[idx])
        image = self.transforms(raw_image)
        if self.one2many:
            texts = [str(self.captions[idx])] + list(self.hard_negs[idx])
        else:
            texts = [str(self.captions[idx])][0]
        return self.images[idx], image, texts

def get_csv_dataset(args, preprocess_fn, is_train):
    input_filename = args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        args, 
        preprocess_fn) 
    num_samples = len(dataset)

    sampler = None
    shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=1,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)
    
def get_data(args, preprocess_val):
    data = {}

    data["val"] = get_csv_dataset(
        args, preprocess_val, is_train=False)
    return data


def evaluate(model, tokenizer, data, args):
    metrics = {}
    id2score = {}
    device = torch.device(args.device)
    model.eval()
    model = model.to(torch.float64).to(device)

    autocast = torch.cuda.amp.autocast
    dataloader = data['val'].dataloader

    # FIXME this does not scale past small eval datasets
    # all_image_features @ all_text_features will blow up memory and compute very quickly
    all_image_features, all_text_features = [], []
    one2many = dataloader.dataset.one2many
    if one2many:
        all_ranks = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            image_id, images, texts = batch
            images = images.to(device)
            texts = [txt[0] for txt in texts]

            if tokenizer!=None: texts = tokenizer(texts).to(device)
            else: texts = texts

            image_emb = model.encode_image(images)
            image_emb /= image_emb.norm(dim = -1, keepdim = True)
            
            text_emb = model.encode_text(texts)
            text_emb /= text_emb.norm(dim = -1, keepdim = True)

            for j in range(image_emb.shape[0]):
                curr_image_emb = image_emb[j:j+1, :]
                curr_text_emb = text_emb[j*6:(j+1)*6, :]
                
                rank = get_one2many_rank(curr_image_emb, curr_text_emb)
                
                all_ranks.append(rank)
                
        
        val_metrics = get_one2many_metrics(np.array(all_ranks))
        metrics.update(
            {**val_metrics}
        )
        
    logging.info("\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()]))
    
    return metrics, id2score

def gather_params(args, hard_neg_type, split):
    if args.compo_type == 'systematicity':
        if hard_neg_type in ['atom', 'comp', 'combined']:
            hard_neg_key = f'valid_hard_negs_{hard_neg_type}'
        else:
            raise NotImplementedError
        
        retrieval_data_path = os.path.join(args.input_dir, f'syst_vg_hard_negs_{split}_in_{args.train_dataset}.csv')
        
    elif args.compo_type == 'productivity':
        hard_neg_key = 'hard_negs'
        if hard_neg_type in ['atom', 'negate', 'swap']:
            input_dir = os.path.join(args.input_dir, hard_neg_type)
            retrieval_data_path = os.path.join(input_dir, f'prod_vg_hard_negs_{hard_neg_type}_complexity_{split}.csv')
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    args.val_data = retrieval_data_path
    args.one2many = True
    args.crop = True
    args.hard_neg_key = hard_neg_key
    args.batch_size = 1
    return args


def default_evaluate(model, transform, tokenizer, type='system'):
    if type == 'system': args = SystemArgs()
    else: args = DefaultArgs()
    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    args.device = device
    device = torch.device(device)
    all_metrics = {}
    for hard_neg_type in args.hard_neg_types:
        for split in args.splits:
            # params = gather_params(args, model, split)
            print('\n' + '*' * 45  + f' Evaluating {args.compo_type} on HN-{hard_neg_type.upper()} test set split {split} ' + '*' * 45  + '\n')
            args = gather_params(args, hard_neg_type, split)
            # initialize datasets
            data = get_data(args, transform)
            assert len(data), 'At least one dataset must be specified.'

            metrics, _ = evaluate(model, tokenizer, data, args)

            if f"{type}_{hard_neg_type}_R@1" not in all_metrics.keys(): all_metrics[f"{type}_{hard_neg_type}_R@1"] = [metrics['image_to_text_R@1']]
            else: all_metrics[f"{type}_{hard_neg_type}_R@1"].append(metrics['image_to_text_R@1'])
    return all_metrics
            

            

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--model', type=str, default="")
    
    parser.add_argument('--pretrained', default=None)
    
    parser.add_argument('--output_folder', default=None)
    
    parser.add_argument('--iter', default=None)
    
    args = parser.parse_args()
    
    model, transform, _ = load_clip(model_type="cust_clip", model_name=args.model, pretrained=args.pretrained,)

    model.eval()
    metrics = {}
    metrics.update(default_evaluate(model, transform, None, type="system"))
    metrics.update(default_evaluate(model, transform, None,  type="prod"))
    
    myDict = {
        "dataset": "crepe",
        "task": "compositionality",
        "metrics": metrics
    }
    
    json_object = json.dumps(myDict, indent=4)
    filePath = os.path.join(args.output_folder, f"{args.iter}_crepe_compositionality.json")
    with open(filePath, "w") as outfile: outfile.write(json_object)

