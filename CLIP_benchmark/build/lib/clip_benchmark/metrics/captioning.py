import json
from open_clip import tokenize
from tqdm.auto import tqdm
from open_clip.tokenizer import _tokenizer


from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice


"""
Code adapted from https://github.com/salaniz/pycocoevalcap/blob/master/eval.py
Thanks to @salaniz for the code!
"""
class COCOEvalCap:
    def __init__(self, results):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.results = results
    def evaluate(self):
        gts = {}
        res = {}
        for imgId, r in enumerate(self.results):
            gts[imgId] = r['true']
            res[imgId] = r['gen']
        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]

import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def reverse_normalize(tensor,):
    """Reverse the normalization of a tensor.
    
    Args:
        tensor (torch.Tensor): The normalized tensor to be reversed.
        mean (tuple): The mean used in normalization.
        std (tuple): The standard deviation used in normalization.
    
    Returns:
        torch.Tensor: The tensor with normalization reversed.
    """
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean




def evaluate(model, dataloader, batch_size, device, transform, train_dataloader=None, num_workers=None, amp=True, verbose=False, pretrained=None):
    results = []
    image_id = 0
    gt = []
    for idx, (img, captions) in enumerate(tqdm(dataloader)):
        out = model.generate(img.to(device))
        decoded = [_tokenizer.decode(i).split("<end_of_text>")[0].replace("<start_of_text>", "").strip() for i in out.cpu().numpy()]
        reversed_img = reverse_normalize(img)
        for i, (pred, true) in enumerate(zip(decoded, captions)):
        
            
            # Load the image
            img = reversed_img[i].cpu().numpy()
            np_image = np.transpose(img, (1, 2, 0))
            np_image = np.clip(np_image, 0, 1) * 255

            # # Create figure and axes
            # fig, ax = plt.subplots(figsize=(8, 10))  # Adjust figure size as needed

            # # Display the image
            # ax.imshow(np_image.astype(np.uint8))
            # ax.axis('off')  # Hide axes

            # # Add captions below the image
            # plt.figtext(0.5, 0.08, "prediction: " + pred, ha="center", fontsize=12, color="blue")
            # plt.figtext(0.5, 0.02, "ground truth: " + true[0], ha="center", fontsize=12, color="green")

            # # Adjust layout and save the figure
            # model_name = pretrained.split("/")[-3]
            # folderpath = f'/gscratch/krishna/chenhaoz/IL/open_clip/captions/{image_id}/{model_name}'
            # os.makedirs(folderpath, exist_ok=True)
            # plt.tight_layout(pad=3.0)
            # plt.savefig(os.path.join(folderpath, f"{os.path.basename(pretrained).split('.')[0]}.png"), dpi=300)  # Adjust DPI as needed
            # plt.clf()
            
            true = [{'caption': t} for t in true]
            pred = [{'caption': pred}]
            results.append({"image_id":image_id, "gen":pred, "true": true})
            
            image_id += 1
    coco_eval = COCOEvalCap(results)
    coco_eval.evaluate()
    metrics = coco_eval.eval
    # print output evaluation scores
    for metric, score in metrics.items():
        print(f'{metric}: {score:.3f}')
    return metrics
