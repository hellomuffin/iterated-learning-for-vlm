from datasets import load_dataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import torch
import json
import argparse
from clip_benchmark.models import MODEL_TYPES, load_clip
from PIL import Image
import torchvision
from torchvision import transforms as TT

image_root = 'datasets/sugar-crepe/data/val2017'
data_root = 'datasets/sugar-crepe/data'
data_dict = {
    'add_obj'    : f'{data_root}/add_obj.json',
    'add_att'    : f'{data_root}/add_att.json',
    'replace_obj': f'{data_root}/replace_obj.json',
    'replace_att': f'{data_root}/replace_att.json',
    'replace_rel': f'{data_root}/replace_rel.json',
    'swap_obj'   : f'{data_root}/swap_obj.json',
    'swap_att'   : f'{data_root}/swap_att.json',
}
dataset = {}
for c, data_path in data_dict.items():
    dataset[c] = json.load(open(data_path, 'r', encoding='utf-8'))


def evaluate(model, transform, output_folder, iter):
    ##------sugar-crepe evaluation
    metrics = {}
    for c, data_dict in dataset.items():
        correct_cnt = 0
        for i, data in tqdm(data_dict.items(), desc=f'evaluating {c}'):
            image_path = os.path.join(image_root, data['filename'])
            image = Image.open(image_path).convert('RGB')
            
            pos_text_embedding = model.encode_text(data['caption'])
            neg_text_embedding = model.encode_text(data['negative_caption'])
            cuda_image = transform(image).unsqueeze(dim=0).cuda()
            image_embedding = model.encode_image(cuda_image)
    
            pos_text_embedding /= pos_text_embedding.norm(dim=-1, keepdim=True)
            neg_text_embedding /= neg_text_embedding.norm(dim=-1, keepdim=True)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            
            pos_score = pos_text_embedding @ image_embedding.t()
            neg_score = neg_text_embedding @ image_embedding.t()
            correct =  1 if pos_score.item() > neg_score.item() else 0
            correct_cnt += correct
            
        count = len(data_dict)
        metrics[c] = correct_cnt / count
    print(metrics)
    myDict = {
        "dataset": "sugar-crepe",
        "task": "compositionality",
        "metrics": metrics
    }
    if output_folder:
        json_object = json.dumps(myDict, indent=4)
        filePath = os.path.join(output_folder, f"{iter}_sugar_crepe_compositionality.json")
        with open(filePath, "w") as outfile: outfile.write(json_object)


                


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--model', type=str, default="")
    
    parser.add_argument('--pretrained', default=None)
    
    parser.add_argument('--output_folder', default=None)
    
    parser.add_argument('--iter', default=None)
    
    args = parser.parse_args()
    
    model, transform, _ = load_clip(model_type="cust_clip", model_name=args.model, pretrained=args.pretrained,)

    evaluate(model, transform, args.output_folder, args.iter)
   