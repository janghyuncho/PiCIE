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




def initialize_classifier(args, n_query, centroids):
    classifier = nn.Conv2d(args.in_dim, n_query, kernel_size=1, stride=1, padding=0, bias=False)
    classifier = nn.DataParallel(classifier)
    classifier = classifier.cuda()
    if centroids is not None:
        classifier.module.weight.data = centroids.unsqueeze(-1).unsqueeze(-1)
    freeze_all(classifier)

    return classifier


def get_testloader(args):
    testset    = EvalDataset(args.data_root, dataset=args.dataset, res=args.res1, split=args.val_type, mode='test', stuff=args.stuff, thing=args.thing)
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=args.batch_size_eval,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             collate_fn=collate_eval)

    return testloader


def compute_dist(featmap, metric_function, euclidean_train=True):
    centroids = metric_function.module.weight.data
    if euclidean_train:
        return - (1 - 2*metric_function(featmap)\
                    + (centroids*centroids).sum(dim=1).unsqueeze(0)) # negative l2 squared 
    else:
        return metric_function(featmap)


def get_nearest_neighbors(n_query, dataloader, model, classifier, k=10):
    model.eval()
    classifier.eval()

    min_dsts = [[] for _ in range(n_query)]
    min_locs = [[] for _ in range(n_query)]
    min_imgs = [[] for _ in range(n_query)]
    with torch.no_grad():
        for indice, image, label in dataloader:
            image = image.cuda(non_blocking=True)
            feats = model(image)
            feats = F.normalize(feats, dim=1, p=2)
            dists = compute_dist(feats, classifier) # (B x C x H x W)
            B, _, H, W = dists.shape
            for c in range(n_query):
                dst, idx = dists[:, c].flatten().topk(1)

                idx = idx.item()
                ib = idx//(H*W)
                ih = idx%(H*W)//W 
                iw = idx%(H*W)%W
                if len(min_dsts[c]) < k:
                    min_dsts[c].append(dst)
                    min_locs[c].append((ib, ih, iw))
                    min_imgs[c].append(indice[ib])
                elif dst < max(min_dsts[c]):
                    imax = np.argmax(min_dsts[c])

                    min_dsts[c] = min_dsts[c][:imax] + min_dsts[c][imax+1:]
                    min_locs[c] = min_locs[c][:imax] + min_locs[c][imax+1:]
                    min_imgs[c] = min_imgs[c][:imax] + min_imgs[c][imax+1:]

                    min_dsts[c].append(dst)
                    min_locs[c].append((ib, ih, iw))
                    min_imgs[c].append(indice[ib])
                
    loclist = min_locs 
    dataset = dataloader.dataset
    imglist = [[dataset.transform_data(*dataset.load_data(dataset.imdb[i]), i, True)[0] for i in ids] for ids in min_imgs]
    return imglist, loclist

if __name__ == '__main__':
    args = parse_arguments()
    
    # Use random seed.
    fix_seed_for_reproducability(args)

    # Init model. 
    model = fpn.PanopticFPN(args)
    model = nn.DataParallel(model)
    model = model.cuda()

    # Load weights.
    checkpoint = torch.load(args.save_root + 'checkpoint.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    
    # Init classifier (for eval only.)
    queries = torch.tensor(np.load('picie_querys.npy')).cuda()
    classifier = initialize_classifier(args, queries.size(0), queries)

    # Prepare testloader.
    testloader = get_testloader(args)

    # Retrieve 10-nearest neighbors.
    imglist, loclist = get_nearest_neighbors(queries.size(0), testloader, model, classifier, k=args.K_test) 

    # Save the result. 
    torch.save([imglist, loclist], args.save_root + '/retrieval_result.pkl')
    print('-Done.', flush=True)

