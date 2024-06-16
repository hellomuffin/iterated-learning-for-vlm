from pathlib import Path
import os
from tqdm import tqdm
import torch
import json
import argparse
from clip_benchmark.models import MODEL_TYPES, load_clip
from PIL import Image
from torch.utils.data import Dataset



class ImageCaptionDataset(Dataset):
    def __init__(self, json_path, images_folder):
        # Load the JSON file
        with open(json_path, 'r') as file:
            self.data = json.load(file)
        
        # Set the images folder
        self.images_folder = Path(images_folder)
        
    
    def __len__(self):
        # Return the number of pairs in the dataset
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get the pair (two images and two captions)
        pair = self.data[idx]
        
        # Load the images and apply transformations
        image_0_path = self.images_folder / Path(pair[0]).name
        image_1_path = self.images_folder / Path(pair[2]).name
        image_0 = Image.open(image_0_path)
        image_1 = Image.open(image_1_path)
        
        # Get the captions
        caption_0 = pair[1]
        caption_1 = pair[3]
        
        # Return the batch
        batch = {
            "image_0": image_0,
            "image_1": image_1,
            "caption_0": caption_0,
            "caption_1": caption_1
        }
        
        return batch
    
cola = ImageCaptionDataset(json_path="data/COLA/data/COLA_multiobjects_matching_benchmark.json", images_folder="data/COLA/images")



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
    for example in tqdm(cola, total=len(cola)):
        images = torch.stack((transform(example["image_0"].convert("RGB")), transform(example["image_1"].convert("RGB"))))
        image_embedding = model.encode_image(images)
        texts = [example["caption_0"], example["caption_1"]]
        text_embedding = model.encode_text(texts)
        logits_per_image, logits_per_text = create_logits(image_embedding, text_embedding, logit_scale=1)
        image_correct, text_correct = group_correct(logits_per_image, logits_per_text)
        text_correct_count += 1 if text_correct else 0
        image_correct_count += 1 if image_correct else 0
        group_correct_count += 1 if text_correct and image_correct else 0
        
    denominator = len(cola)
    metrics = {
        "text_score": text_correct_count/denominator,
        "image_score": image_correct_count/denominator,
        "group_score": group_correct_count/denominator
    }
    print(metrics)
    myDict = {
        "dataset": "cola",
        "task": "compositionality",
        "metrics": metrics
    }
    
    json_object = json.dumps(myDict, indent=4)
    filePath = os.path.join(output_folder, f"{iter}_cola_compositionality.json")
    with open(filePath, "w") as outfile: outfile.write(json_object)

# # Serializing json
# json_object = json.dumps(dictionary, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classification Solver')
    parser.add_argument('--model', type=str, default="")
    
    parser.add_argument('--pretrained', default=None)
    
    parser.add_argument('--output_folder', default=None)
    
    parser.add_argument('--iter', default=None)
    
    args = parser.parse_args()
    
    model, transform, _ = load_clip(model_type="cust_clip", model_name=args.model, pretrained=args.pretrained,)

    evaluate(model, transform, args.output_folder, args.iter)
   