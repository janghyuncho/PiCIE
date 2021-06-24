import os 
import torch 
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np 
from PIL import Image, ImageFilter
import json
import random 
import cv2
import pickle

FINE_TO_COARSE_PATH = 'fine_to_coarse_dict.pickle'

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class EvalCOCO(data.Dataset):
    def __init__(self, root, split, mode, res=128, transform_list=[], label=True, stuff=True, thing=False):
        self.root  = root 
        self.split = split
        self.mode  = mode
        self.res   = res 
        self.imdb  = self.load_imdb()
        self.stuff = stuff 
        self.thing = thing 
        self.label = label
        self.view  = -1

        self.fine_to_coarse = self._get_fine_to_coarse()

        # For test-time augmentation / robustness test. 
        self.transform_list = transform_list
        
    def load_imdb(self):
        # 1. Setup filelist
        imdb = os.path.join(self.root, 'curated', '{}2017'.format(self.split), 'Coco164kFull_Stuff_Coarse_7.txt')
        imdb = tuple(open(imdb, "r"))
        imdb = [id_.rstrip() for id_ in imdb]
        
        return imdb
    
    def __getitem__(self, index):
        image_id = self.imdb[index]
        img, lbl = self.load_data(image_id)

        return (index,) + self.transform_data(img, lbl, index)

    def load_data(self, image_id):
        """
        Labels are in unit8 format where class labels are in [0 - 181] and 255 is unlabeled.
        """
        N = len(self.imdb)
        image_path = os.path.join(self.root, 'images', '{}2017'.format(self.split), '{}.jpg'.format(image_id))
        label_path = os.path.join(self.root, 'annotations', '{}2017'.format(self.split), '{}.png'.format(image_id))

        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path)

        return image, label

    def transform_data(self, image, label, index):

        # 1. Resize
        image = TF.resize(image, self.res, Image.BILINEAR)
        label = TF.resize(label, self.res, Image.NEAREST)
        
        # 2. CenterCrop
        w, h = image.size
        left = int(round((w - self.res) / 2.))
        top  = int(round((h - self.res) / 2.))

        image = TF.crop(image, top, left, self.res, self.res)
        label = TF.crop(label, top, left, self.res, self.res)

        # 3. Transformation
        image = self._image_transform(image, self.mode)
        if not self.label:
            return (image, None)

        label = self._label_transform(label)

        return image, label


    def _get_fine_to_coarse(self):
        """
        Map fine label indexing to coarse label indexing. 
        """
        with open(os.path.join(self.root, FINE_TO_COARSE_PATH), "rb") as dict_f:
            d = pickle.load(dict_f)
        fine_to_coarse_dict      = d["fine_index_to_coarse_index"]
        fine_to_coarse_dict[255] = -1
        fine_to_coarse_map       = np.vectorize(lambda x: fine_to_coarse_dict[x]) # not in-place.

        return fine_to_coarse_map


    def _label_transform(self, label):
        """
        In COCO-Stuff, there are 91 Things and 91 Stuff. 
            91 Things (0-90)  => 12 superclasses (0-11)
            91 Stuff (91-181) => 15 superclasses (12-26)

        For [Stuff-15], which is the benchmark IIC uses, we only use 15 stuff superclasses.
        """
        label = np.array(label)
        label = self.fine_to_coarse(label)    # Map to superclass indexing.
        mask  = label >= 255 # Exclude unlabelled.
        
        # Start from zero. 
        if self.stuff and not self.thing:
            label[mask] -= 12 # This makes all Things categories negative (ignored.)
        elif self.thing and not self.stuff:
            mask = label > 11 # This makes all Stuff categories negative (ignored.)
            label[mask] = -1
            
        # Tensor-fy
        label = torch.LongTensor(label)                            

        return label


    def _image_transform(self, image, mode):
        if self.mode == 'test':
            transform = self._get_data_transformation()

            return transform(image)
        else:
            raise NotImplementedError()


    def _get_data_transformation(self):
        trans_list = []
        if 'jitter' in self.transform_list:
            trans_list.append(transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)], p=0.8))
        if 'grey' in self.transform_list:
            trans_list.append(transforms.RandomGrayscale(p=0.2))
        if 'blur' in self.transform_list:
            trans_list.append(transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5))
        
        # Base transformation
        trans_list += [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

        return transforms.Compose(trans_list)
    
    def __len__(self):
        return len(self.imdb)
        

  
            
       
