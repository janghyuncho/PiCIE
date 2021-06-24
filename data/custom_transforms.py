import torch 
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np 
from PIL import Image, ImageFilter
import random 

class BaseTransform(object):
    """
    Resize and center crop. 
    """
    def __init__(self, res):
        self.res = res 
        
    def __call__(self, index, image):
        image = TF.resize(image, self.res, Image.BILINEAR)
        w, h  = image.size
        left  = int(round((w - self.res) / 2.))
        top   = int(round((h - self.res) / 2.))

        return TF.crop(image, top, left, self.res, self.res)


class ComposeTransform(object):
    def __init__(self, tlist):
        self.tlist = tlist 
    
    def __call__(self, index, image):
        for trans in self.tlist:
            image = trans(index, image)
        
        return image 

class RandomResize(object):
    def __init__(self, rmin, rmax, N):
        self.reslist = [random.randint(rmin, rmax) for _ in range(N)]

    def __call__(self, index, image):
        return TF.resize(image, self.reslist[index], Image.BILINEAR)

class RandomCrop(object):
    def __init__(self, res, N):
        self.res  = res 
        self.cons = [(np.random.uniform(0, 1), np.random.uniform(0, 1)) for _ in range(N)] 
    
    def __call__(self, index, image):
        ws, hs = self.cons[index]
        w, h   = image.size
        left = int(round((w-self.res)*ws))
        top  = int(round((h-self.res)*hs))

        return TF.crop(image, top, left, self.res, self.res)

class RandomHorizontalFlip(object):
    def __init__(self, N, p=0.5):
        self.p_ref = p
        self.plist = np.random.random_sample(N)

    def __call__(self, index, image):
        if self.plist[index] < self.p_ref:
            return TF.hflip(image)
        else:
            return image


class TensorTransform(object):
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, image):
        image = self.to_tensor(image)
        image = self.normalize(image)

        return image
        
        
class RandomGaussianBlur(object):
    def __init__(self, sigma, p, N):
        self.min_x = sigma[0]
        self.max_x = sigma[1]
        self.del_p = 1 - p
        self.p_ref = p
        self.plist = np.random.random_sample(N)

    def __call__(self, index, image):
        if self.plist[index] < self.p_ref:
            x = self.plist[index] - self.p_ref 
            m = (self.max_x - self.min_x) / self.del_p
            b = self.min_x 
            s = m * x + b 

            return image.filter(ImageFilter.GaussianBlur(radius=s))
        else:
            return image


class RandomGrayScale(object):
    def __init__(self, p, N):
        self.grayscale = transforms.RandomGrayscale(p=1.) # Deterministic (We still want flexible out_dim).
        self.p_ref = p
        self.plist = np.random.random_sample(N)

    def __call__(self, index, image):
        if self.plist[index] < self.p_ref:
            return self.grayscale(image)
        else:
            return image


class RandomColorBrightness(object):
    def __init__(self, x, p, N):
        self.min_x = max(0, 1 - x)
        self.max_x = 1 + x
        self.p_ref = p
        self.plist = np.random.random_sample(N)
        self.rlist = [random.uniform(self.min_x, self.max_x) for _ in range(N)]

    def __call__(self, index, image):
        if self.plist[index] < self.p_ref:
            return TF.adjust_brightness(image, self.rlist[index])
        else:
            return image


class RandomColorContrast(object):
    def __init__(self, x, p, N):
        self.min_x = max(0, 1 - x)
        self.max_x = 1 + x
        self.p_ref = p
        self.plist = np.random.random_sample(N)
        self.rlist = [random.uniform(self.min_x, self.max_x) for _ in range(N)]

    def __call__(self, index, image):
        if self.plist[index] < self.p_ref:
            return TF.adjust_contrast(image, self.rlist[index])
        else:
            return image


class RandomColorSaturation(object):
    def __init__(self, x, p, N):
        self.min_x = max(0, 1 - x)
        self.max_x = 1 + x
        self.p_ref = p
        self.plist = np.random.random_sample(N)
        self.rlist = [random.uniform(self.min_x, self.max_x) for _ in range(N)]

    def __call__(self, index, image):
        if self.plist[index] < self.p_ref:
            return TF.adjust_saturation(image, self.rlist[index])
        else:
            return image


class RandomColorHue(object):
    def __init__(self, x, p, N):
        self.min_x = -x
        self.max_x = x
        self.p_ref = p
        self.plist = np.random.random_sample(N)
        self.rlist = [random.uniform(self.min_x, self.max_x) for _ in range(N)]

    def __call__(self, index, image):
        if self.plist[index] < self.p_ref:
            return TF.adjust_hue(image, self.rlist[index])
        else:
            return image


class RandomVerticalFlip(object):
    def __init__(self, N, p=0.5):
        self.p_ref = p
        self.plist = np.random.random_sample(N)
        
    def __call__(self, indice, image):
        I = np.nonzero(self.plist[indice] < self.p_ref)[0]

        if len(image.size()) == 3:
            image_t = image[I].flip([1]) 
        else:
            image_t = image[I].flip([2])
        
        return torch.stack([image_t[np.where(I==i)[0][0]] if i in I else image[i] for i in range(image.size(0))])



class RandomHorizontalTensorFlip(object):
    def __init__(self, N, p=0.5):
        self.p_ref = p
        self.plist = np.random.random_sample(N)

    def __call__(self, indice, image, is_label=False):
        I = np.nonzero(self.plist[indice] < self.p_ref)[0]
        
        if len(image.size()) == 3:
            image_t = image[I].flip([2])
        else:
            image_t = image[I].flip([3])
        
        return torch.stack([image_t[np.where(I==i)[0][0]] if i in I else image[i] for i in range(image.size(0))])



class RandomResizedCrop(object):
    def __init__(self, N, res, scale=(0.5, 1.0)):
        self.res    = res
        self.scale  = scale 
        self.rscale = [np.random.uniform(*scale) for _ in range(N)]
        self.rcrop  = [(np.random.uniform(0, 1), np.random.uniform(0, 1)) for _ in range(N)]

    def random_crop(self, idx, img):
        ws, hs = self.rcrop[idx]
        res1 = int(img.size(-1))
        res2 = int(self.rscale[idx]*res1)
        i1 = int(round((res1-res2)*ws))
        j1 = int(round((res1-res2)*hs))

        return img[:, :, i1:i1+res2, j1:j1+res2]


    def __call__(self, indice, image):
        new_image = []
        res_tar   = self.res // 4 if image.size(1) > 5 else self.res # View 1 or View 2? 
        
        for i, idx in enumerate(indice):
            img = image[[i]]
            img = self.random_crop(idx, img)
            img = F.interpolate(img, res_tar, mode='bilinear', align_corners=False)

            new_image.append(img)

        new_image = torch.cat(new_image)
        
        return new_image



            
            
            
            




    
