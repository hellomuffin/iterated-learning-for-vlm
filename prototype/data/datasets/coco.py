import os
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt



def coco_transform(image, mask):
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.ToTensor(),
    ])
    image = transform(image)
    mask = transform(mask)
    return image, mask

class COCODataset(Dataset):
    def __init__(self, root_dir, annotations_file, save_path, mode='image'):
        self.root_dir = root_dir
        self.coco = COCO(annotations_file)
        self.ann_ids = list(self.coco.anns.keys())
        self.img_ids = list(self.coco.imgs.keys())
        self.save_path = save_path
        self.mode = mode

    def __len__(self):
        if self.mode == 'image':
            return len(self.img_ids)
        elif self.mode == 'annotation':
            return len(self.ann_ids)
        else:
            raise NotImplementedError

    
    
    def __getitem__(self, index):
        if self.mode == 'annotation':
            ann_id = self.ann_ids[index]
            mask, category_name, img_id = self.get_mask_from_annotation(ann_id)
            maskpil = Image.fromarray((mask * 255).astype(np.uint8))
            img_info = self.coco.loadImgs(img_id)[0]
            imagepil = Image.open(os.path.join(self.root_dir, img_info['file_name'])).convert("RGB")
            return maskpil, imagepil, category_name, img_info
        elif self.mode == 'image':
            img_info = self.coco.loadImgs(self.img_ids[index])[0]
            imagepil = Image.open(os.path.join(self.root_dir, img_info['file_name'])).convert("RGB")
            background_mask = self.get_background_mask(index)
            maskpil = Image.fromarray((background_mask * 255).astype(np.uint8))
            return imagepil, maskpil, img_info
        else: raise NotImplementedError
            
    def get_image_by_index(self, index):
        img_info = self.coco.loadImgs(self.img_ids[index])[0]
        imagepil = Image.open(os.path.join(self.root_dir, img_info['file_name'])).convert("RGB")
        return imagepil
    
    def get_background_mask(self, img_id):
        img_info = self.coco.loadImgs(img_id)[0]
        img_height, img_width = img_info['height'], img_info['width']

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        background_mask = np.ones((img_height, img_width), dtype=np.bool)

        for ann in anns:
            object_mask = self.coco.annToMask(ann)
            background_mask &= ~object_mask

        return background_mask

    
    def get_mask_from_annotation(self, ann_id):
        ann = self.coco.loadAnns(ann_id)[0]
        img_id = ann['image_id']
        category_id = ann['category_id']
        category_info = self.coco.loadCats(category_id)[0]
        category_name = category_info['name']
        mask = self.coco.annToMask(ann)
        return mask, category_name, img_id

    def get_supercategory(self, category_name=None, category_id=None):
        if category_name is not None:
            category_id = self.coco.getCatIds(catNms=[category_name])[0]
        category_info = self.coco.loadCats(category_id)[0]
        supercategory = category_info['supercategory']
        return supercategory

    def get_categories_by_supercategory(self, supercategory_name):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories_in_supercategory = [cat['name'] for cat in categories if cat['supercategory'] == supercategory_name]
        return categories_in_supercategory

    def get_all_supercategories(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        supercategories = sorted(set(cat['supercategory'] for cat in categories))
        return supercategories
    
    
    def test_and_save_images(self, index):
            ann_id = self.ann_ids[index]
            mask, category_name, img_id = self.get_mask_from_annotation(ann_id)
            
            mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
            
            img_info = self.coco.loadImgs(img_id)[0]
            img_path = os.path.join(self.root_dir, img_info['file_name'])
            image = Image.open(img_path).convert("RGB")
            
            # Create a 2x2 grid of subplots
            fig, axes = plt.subplots(2, 1, figsize=(200, 100))

            # Add images to the subplots
            ax = axes[0]
            ax.imshow(mask_image)
            ax.axis('off')
            
            ax = axes[1]
            ax.imshow(image)
            ax.axis('off')
            
            output_path = os.path.join(self.save_path, 'test', f'{index}.jpg')
            plt.savefig(output_path, bbox_inches='tight')
            
            return 
    
import random

class COCOCaptionDataset(COCODataset):
    def __init__(self, root_dir, annotations_file, save_path, transform=None, tokenzier=None, mode='image', all_caption=False):
        self.transform = transform
        self.tokenizer = tokenzier
        self.all_caption = all_caption
        super().__init__(root_dir, annotations_file, save_path, mode)
    
    def __getitem__(self, index):
        img_info = self.coco.loadImgs(self.img_ids[index])[0]
        imagepil = Image.open(os.path.join(self.root_dir, img_info['file_name'])).convert("RGB")
        captions = self.get_image_captions(self.img_ids[index])
        if not self.all_caption: captions = [random.choice(captions)]
        
        if self.transform:
            image = self.transform(imagepil)
        if self.tokenizer:
            t_caption = self.tokenizer(captions)
            if not self.all_caption: t_caption = t_caption[0]
            else: raise NotImplementedError
        else: t_caption = captions
        return image, t_caption
    
    def get_image_captions(self, img_id):
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        captions = [ann['caption'] for ann in anns]
        return captions
