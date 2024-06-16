from datasets import load_dataset
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import torch
import json
import argparse
from clip_benchmark.models import MODEL_TYPES, load_clip
from PIL import Image
import torch.nn as nn


auth_token = ""  # TODO: Replace with an auth token, which you can get from your huggingface account: Profile -> Settings -> Access Tokens -> New Token
winoground = load_dataset("facebook/winoground", use_auth_token=auth_token, data_dir="data", cache_dir="data")["test"]



def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 =  logit_scale * x1 @ x2.t()  
    logits_per_x2 =  logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2
    


def compute_correct(result):
    return result[0,0] > result[1,0] and result[1,1] > result[0,1]


def group_correct(logits_per_image, logits_per_text):
    return compute_correct(logits_per_image,), compute_correct(logits_per_text)

def evaluate(model, transform, output_folder, iter):
    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    for example in tqdm(winoground, total=len(winoground)):
        images = torch.stack((transform(example["image_0"].convert("RGB")), transform(example["image_1"].convert("RGB"))))
        image_embedding = model.encode_image(images)
        texts = [example["caption_0"], example["caption_1"]]
        text_embedding = model.encode_text(texts)
        logits_per_image, logits_per_text = create_logits(image_embedding, text_embedding, logit_scale=1)
        image_correct, text_correct = group_correct(logits_per_image, logits_per_text)
        text_correct_count += 1 if text_correct else 0
        image_correct_count += 1 if image_correct else 0
        group_correct_count += 1 if text_correct and image_correct else 0
        
    denominator = len(winoground)
    metrics = {
        "text_score": text_correct_count/denominator,
        "image_score": image_correct_count/denominator,
        "group_score": group_correct_count/denominator
    }
    print(metrics)
    myDict = {
        "dataset": "winoground",
        "task": "compositionality",
        "metrics": metrics
    }
    
    json_object = json.dumps(myDict, indent=4)
    filePath = os.path.join(output_folder, f"{iter}_winoground_compositionality.json")
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
    
    