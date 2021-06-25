import os 
import sys 
import argparse 
import logging
import time as t 
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
from module import fpn
from commons import *
from utils import *
from train_picie import * 

def compute_dist(featmap, metric_function, euclidean_train=True):
    centroids = metric_function.module.weight.data
    if euclidean_train:
        return - (1 - 2*metric_function(featmap)\
                    + (centroids*centroids).sum(dim=1).unsqueeze(0)) # negative l2 squared 
    else:
        return metric_function(featmap)


def compute_histogram(args, dataloader, model, classifier):
    histogram = np.zeros((args.K_test, args.K_test))

    model.eval()
    classifier.eval()
    with torch.no_grad():
        for i, (indice, image, label) in enumerate(dataloader):
            image = image.cuda(non_blocking=True)
            feats = model(image)
            feats = F.normalize(feats, dim=1, p=2)

            if i == 0:
                print('Batch image size   : {}'.format(image.size()), flush=True)
                print('Batch label size   : {}'.format(label.size()), flush=True)
                print('Batch feature size : {}\n'.format(feats.size()), flush=True)
            
            probs = compute_dist(feats, classifier)
            probs = F.interpolate(probs, args.res2, mode='bilinear', align_corners=False)
            preds = probs.topk(1, dim=1)[1].view(probs.size(0), -1).cpu().numpy()
            label = label.view(probs.size(0), -1).cpu().numpy()

            histogram += get_score_histogram(label, preds, args.K_test)
            
    return histogram

if __name__ == '__main__':
    args = get_args()
    
    # Use random seed.
    fix_seed_for_reproducability(args)

    # Init model. 
    model = fpn.PanopticFPN(args)
    model = nn.DataParallel(model)
    model = model.cuda()

    # Init classifier (for eval only.)
    classifier = initialize_classifier(args, args.K_test)

    # Load weights.
    checkpoint = torch.load(args.save_root + 'checkpoint.pth.tar') # Place the checkpoint.pth.tar in the directory.
    model.load_state_dict(checkpoint['state_dict'])
    classifier.load_state_dict(checkpoint['classifier1_state_dict'])

    # Prepare testloader.
    trainloader, validloader = get_testloader(args)

    # Compute statistics.
    histogram_train = compute_histogram(args, trainloader, model, classifier)
    histogram_valid = compute_histogram(args, validloader, model, classifier)

    # Save the result. 
    torch.save([histogram_train, histogram_valid], args.save_root + '/histogram_coco.pkl')
    print('-Done.', flush=True)

