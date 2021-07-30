import os 
import torch 
import torch.nn as nn 
import torch.utils.data as data
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np 
from PIL import Image, ImageFilter
from data.custom_transforms import *
        

class TrainCOCO(data.Dataset):
    def __init__(self, root, labeldir, mode, split='train', res1=320, res2=640, inv_list=[], eqv_list=[], \
                 stuff=True, thing=False, scale=(0.5, 1), version=7, fullcoco=False):
        self.root  = root 
        self.split = split
        self.res1  = res1
        self.res2  = res2  
        self.stuff = stuff 
        self.thing = thing 
        self.mode  = mode
        self.scale = scale 
        self.view  = -1

        self.inv_list = inv_list
        self.eqv_list = eqv_list
        self.labeldir = labeldir

        self.version  = version  # 7 is what we used. 
        self.fullcoco = fullcoco # To use full set of images in COCO.
        
        self.imdb = self.load_imdb()
        self.reshuffle() 

    def load_imdb(self):
        if self.fullcoco:
            imdb = [x[:-4] for x in os.listdir(os.path.join(self.root, '{}2017'.format(self.split)))]
        else:
            # Setup filelist for the main benchmark. (This will have the same set of images as IIC.)
            # https://github.com/xu-ji/IIC/blob/master/code/datasets/segmentation/cocostuff.py
            # https://github.com/xu-ji/IIC/blob/master/examples/commands.txt (version 7 was used here).
            imdb = os.path.join(self.root, 'curated', '{}2017'.format(self.split), 'Coco164kFull_Stuff_Coarse_{}.txt'.format(self.version))
            imdb = tuple(open(imdb, "r"))
            imdb = [id_.rstrip() for id_ in imdb]
            
        return imdb
    

    def __getitem__(self, index):
        index = self.shuffled_indices[index]
        imgid = self.imdb[index]
        image = self.load_data(imgid)

        image = self.transform_image(index, image)
        label = self.transform_label(index)
        
        return (index, ) + image + label


    def load_data(self, image_id):
        """
        Labels are in unit8 format where class labels are in [0 - 181] and 255 is unlabeled.
        """
        image_path = os.path.join(self.root, 'images', '{}2017'.format(self.split), '{}.jpg'.format(image_id))

        return Image.open(image_path).convert('RGB')


    def reshuffle(self):
        """
        Generate random floats for all image data to deterministically random transform.
        This is to use random sampling but have the same samples during clustering and 
        training within the same epoch. 
        """
        self.shuffled_indices = np.arange(len(self.imdb))
        np.random.shuffle(self.shuffled_indices)
        self.init_transforms()


    def transform_image(self, index, image):
        # Base transform
        image = self.transform_base(index, image)

        if self.mode == 'compute':
            if self.view == 1:
                image = self.transform_inv(index, image, 0)
                image = self.transform_tensor(image)
            elif self.view == 2:
                image = self.transform_inv(index, image, 1)
                image = TF.resize(image, self.res1, Image.BILINEAR)
                image = self.transform_tensor(image)
            else:
                raise ValueError('View [{}] is an invalid option.'.format(self.view))
            return (image, )
        elif 'train' in self.mode:
            # Invariance transform. 
            image1 = self.transform_inv(index, image, 0)
            image1 = self.transform_tensor(image1)

            if self.mode == 'baseline_train':
                return (image1, )
            
            image2 = self.transform_inv(index, image, 1)
            image2 = TF.resize(image2, self.res1, Image.BILINEAR)
            image2 = self.transform_tensor(image2)

            return (image1, image2)
        else:
            raise ValueError('Mode [{}] is an invalid option.'.format(self.mode))


    def transform_inv(self, index, image, ver):
        """
        Hyperparameters same as MoCo v2. 
        (https://github.com/facebookresearch/moco/blob/master/main_moco.py)
        """
        if 'brightness' in self.inv_list:
            image = self.random_color_brightness[ver](index, image)
        if 'contrast' in self.inv_list:
            image = self.random_color_contrast[ver](index, image)
        if 'saturation' in self.inv_list:
            image = self.random_color_saturation[ver](index, image)
        if 'hue' in self.inv_list:
            image = self.random_color_hue[ver](index, image)
        if 'gray' in self.inv_list:
            image = self.random_gray_scale[ver](index, image)
        if 'blur' in self.inv_list:
            image = self.random_gaussian_blur[ver](index, image)
        
        return image



    def transform_eqv(self, indice, image):
        if 'random_crop' in self.eqv_list:
            image = self.random_resized_crop(indice, image)
        if 'h_flip' in self.eqv_list:
            image = self.random_horizontal_flip(indice, image)
        if 'v_flip' in self.eqv_list:
            image = self.random_vertical_flip(indice, image)

        return image


    def init_transforms(self):
        N = len(self.imdb)
        
        # Base transform.
        self.transform_base = BaseTransform(self.res2)
        
        # Transforms for invariance. 
        # Color jitter (4), gray scale, blur. 
        self.random_color_brightness = [RandomColorBrightness(x=0.3, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)]
        self.random_color_contrast   = [RandomColorContrast(x=0.3, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)
        self.random_color_saturation = [RandomColorSaturation(x=0.3, p=0.8, N=N) for _ in range(2)] # Control this later (NOTE)
        self.random_color_hue        = [RandomColorHue(x=0.1, p=0.8, N=N) for _ in range(2)]      # Control this later (NOTE)
        self.random_gray_scale    = [RandomGrayScale(p=0.2, N=N) for _ in range(2)]
        self.random_gaussian_blur = [RandomGaussianBlur(sigma=[.1, 2.], p=0.5, N=N) for _ in range(2)]

        self.random_horizontal_flip = RandomHorizontalTensorFlip(N=N)
        self.random_vertical_flip   = RandomVerticalFlip(N=N)
        self.random_resized_crop    = RandomResizedCrop(N=N, res=self.res1, scale=self.scale)

        # Tensor transform. 
        self.transform_tensor = TensorTransform()
    

    def transform_label(self, index):
        # TODO Equiv. transform.
        if self.mode == 'train':
            label1 = torch.load(os.path.join(self.labeldir, 'label_1', '{}.pkl'.format(index)))
            label2 = torch.load(os.path.join(self.labeldir, 'label_2', '{}.pkl'.format(index)))
            label1 = torch.LongTensor(label1)
            label2 = torch.LongTensor(label2)

            X1 = int(np.sqrt(label1.shape[0]))
            X2 = int(np.sqrt(label2.shape[0]))
            
            label1 = label1.view(X1, X1)
            label2 = label2.view(X2, X2)

            return label1, label2

        elif self.mode == 'baseline_train':
            label1 = torch.load(os.path.join(self.labeldir, 'label_1', '{}.pkl'.format(index)))
            label1 = torch.LongTensor(label1)

            X1 = int(np.sqrt(label1.shape[0]))
            
            label1 = label1.view(X1, X1)

            return (label1, )

        return (None, )


    def __len__(self):
        return len(self.imdb)
        

  
            
       
