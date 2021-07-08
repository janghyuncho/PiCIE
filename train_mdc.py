import argparse
import os
import time as t
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.coco_train_dataset import TrainCOCO
from data.coco_eval_dataset import EvalCOCO 
from utils import *
from commons import *
from modules import fpn 

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--save_root', type=str, required=True)
    parser.add_argument('--restart_path', type=str)
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--seed', type=int, default=2021, help='Random seed for reproducability.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers.')
    parser.add_argument('--restart', action='store_true', default=False)
    parser.add_argument('--num_epoch', type=int, default=10) 
    parser.add_argument('--repeats', type=int, default=10)  

    # Train. 
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--res', type=int, default=320, help='Input size.')
    parser.add_argument('--res1', type=int, default=320, help='Input size scale from.')
    parser.add_argument('--res2', type=int, default=320, help='Input size scale to.')
    parser.add_argument('--batch_size_cluster', type=int, default=128)
    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optim_type', type=str, default='Adam')
    parser.add_argument('--num_init_batches', type=int, default=30)
    parser.add_argument('--num_batches', type=int, default=30)
    parser.add_argument('--kmeans_n_iter', type=int, default=30)
    parser.add_argument('--in_dim', type=int, default=128)
    parser.add_argument('--X', type=int, default=80)
    parser.add_argument('--nonparametric', action='store_true', default=False)

    # Loss. 
    parser.add_argument('--metric_train', type=str, default='cosine')   
    parser.add_argument('--metric_test', type=str, default='cosine')
    parser.add_argument('--K_train', type=int, default=1000) # COCO Stuff-15
    parser.add_argument('--K_test', type=int, default=27) # COCO Stuff-15 / COCO Thing-12 / COCO All-27

    # Dataset. 
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--equiv', action='store_true', default=False)
    parser.add_argument('--min_scale', type=float, default=0.5)
    parser.add_argument('--stuff', action='store_true', default=False)
    parser.add_argument('--thing', action='store_true', default=False)
    parser.add_argument('--jitter', action='store_true', default=False)
    parser.add_argument('--grey', action='store_true', default=False)
    parser.add_argument('--blur', action='store_true', default=False)
    parser.add_argument('--h_flip', action='store_true', default=False)
    parser.add_argument('--v_flip', action='store_true', default=False)
    parser.add_argument('--random_crop', action='store_true', default=False)
    parser.add_argument('--val_type', type=str, default='val')
    parser.add_argument('--version', type=int, default=7)
    parser.add_argument('--fullcoco', action='store_true', default=False)
    
    # Eval-only
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_path', type=str)

    return parser.parse_args()



def train(args, logger, dataloader, model, classifier, criterion, optimizer, optimizer_loop):
    losses = AverageMeter()

    # switch to train mode
    model.train()
    classifier.train(not args.nonparametric)
    for i, (indice, image, label) in enumerate(dataloader):
        image = eqv_transform_if_needed(args, dataloader, indice, image.cuda(non_blocking=True))
        label = label.cuda(non_blocking=True)
        feats = model(image)

        B, C, _ = feats.shape[:3]
        if i == 0:
            logger.info('Batch input size   : {}'.format(list(image.shape)))
            logger.info('Batch label size   : {}'.format(list(label.shape)))
            logger.info('Batch feature size : {}\n'.format(list(feats.shape)))
        
        if args.metric_train == 'cosine':
            feats = F.normalize(feats, dim=1, p=2)
        
        # 2. Get scores.
        output = feature_flatten(classifier(feats))
        label  = label.flatten()
        loss   = criterion(output, label)
        
        # record loss
        losses.update(loss.item(), B)

        # compute gradient and do step
        optimizer.zero_grad()
        if not args.nonparametric:
            optimizer_loop.zero_grad()
        loss.backward()
        optimizer.step()
        if not args.nonparametric:
            optimizer_loop.step()

        if (i % 200) == 0:
            logger.info('{0} / {1}\t'.format(i, len(dataloader)))

    return losses.avg



def main(args, logger):
    logger.info(args)

    # Use random seed.
    fix_seed_for_reproducability(args.seed)

    # Start time.
    t_start = t.time()

    # Get model and optimizer.
    model, optimizer, classifier = get_model_and_optimizer(args, logger)

    # New trainset inside for-loop.
    inv_list, eqv_list = get_transform_params(args)
    trainset = TrainCOCO(args.data_root, res1=args.res1, res2=args.res2,\
                        split='train', mode='compute', labeldir='', inv_list=inv_list, eqv_list=eqv_list, stuff=args.stuff,\
                        thing=args.thing,\
                        scale=(args.min_scale, 1)) # NOTE: For now, max_scale = 1.  
    trainloader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=args.batch_size_cluster,
                                                shuffle=True, 
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                collate_fn=collate_train_baseline,
                                                worker_init_fn=worker_init_fn(args.seed))

    testset    = EvalCOCO(args.data_root, res=args.res, split='val', mode='test', stuff=args.stuff, thing=args.thing)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size_test,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             collate_fn=collate_eval,
                                             worker_init_fn=worker_init_fn(args.seed))

    # Train start.
    for epoch in range(args.start_epoch, args.num_epoch):
        # Assign probs. 
        trainloader.dataset.mode = 'compute'
        trainloader.dataset.reshuffle()

        logger.info('\n============================= [Epoch {}] =============================\n'.format(epoch))
        logger.info('Start computing centroids.')
        t1 = t.time()
        centroids, kmloss = run_mini_batch_kmeans(args, logger, trainloader, model, view=1)
        logger.info('-Centroids ready. [{}]\n'.format(get_datetime(int(t.time())-int(t1))))

        # Compute cluster assignment. 
        t2 = t.time()
        weight = compute_labels(args, logger, trainloader, model, centroids, view=1) 
        logger.info('-Cluster labels ready. [{}]\n'.format(get_datetime(int(t.time())-int(t2)))) 

        # Criterion. 
        criterion = torch.nn.CrossEntropyLoss(weight=weight).cuda()

        # Set nonparametric classifier.
        classifier = initialize_classifier(args)
        if args.nonparametric:
            classifier.module.weight.data = centroids.unsqueeze(-1).unsqueeze(-1)
            freeze_all(classifier)

        if args.nonparametric:
            optimizer_loop = None 
        else:
            if args.optim_type == 'SGD':
                optimizer_loop = torch.optim.SGD(filter(lambda x: x.requires_grad, classifier.module.parameters()), lr=args.lr, \
                                                  momentum=args.momentum, weight_decay=args.weight_decay)
            elif args.optim_type == 'Adam':
                optimizer_loop = torch.optim.Adam(filter(lambda x: x.requires_grad, classifier.module.parameters()), lr=args.lr)


        # Set-up train loader.
        trainset.mode  = 'baseline_train'
        trainset.labeldir = args.save_model_path
        trainloader_loop  = torch.utils.data.DataLoader(trainset, 
                                                        batch_size=args.batch_size_train, 
                                                        shuffle=True,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True,
                                                        collate_fn=collate_train_baseline,
                                                        worker_init_fn=worker_init_fn(args.seed)
                                                        )

        logger.info('Start training ...')
        train_loss = train(args, logger, trainloader_loop, model, classifier, criterion, optimizer, optimizer_loop) 
        acc, res   = evaluate(args, logger, testloader, classifier, model)

        logger.info('========== Epoch [{}] =========='.format(epoch))
        logger.info('  Time total : [{}].'.format(get_datetime(int(t.time())-int(t1))))
        logger.info('  K-Means loss   : {:.5f}.'.format(kmloss))
        logger.info('  Training loss  : {:.5f}.'.format(train_loss))
        logger.info('  ACC: {:.4f} | mIoU: {:.4f}'.format(acc, res['mean_iou']))
        logger.info('================================\n')

        torch.save({'epoch': epoch+1, 
                    'args' : args,
                    'state_dict': model.state_dict(),
                    'classifier1_state_dict' : classifier.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    },
                    os.path.join(args.save_model_path, 'checkpoint_{}.pth.tar'.format(epoch)))
        
        torch.save({'epoch': epoch+1, 
                    'args' : args,
                    'state_dict': model.state_dict(),
                    'classifier1_state_dict' : classifier.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    },
                    os.path.join(args.save_model_path, 'checkpoint.pth.tar'))
    
    # Evaluate.
    trainset    = TrainCOCO(args.data_root, res=args.res, split=args.val_type, mode='test', label=False) 
    trainloader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=args.batch_size_cluster,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                collate_fn=collate_train_baseline,
                                                worker_init_fn=worker_init_fn(args.seed)
                                                )

    testset    = EvalCOCO(args.data_root, res=args.res, split='val', mode='test', stuff=args.stuff, thing=args.thing)
    testloader = torch.utils.data.DataLoader(testset, 
                                             batch_size=args.batch_size_test,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             collate_fn=collate_eval,
                                             worker_init_fn=worker_init_fn(args.seed))
    # Evaluate with fresh clusters. 
    acc_list_new = []  
    res_list_new = []                 
    logger.info('Start computing centroids.')
    if args.repeats > 0:
        for _ in range(args.repeats):
            t1 = t.time()
            centroids, kmloss = run_mini_batch_kmeans(args, logger, trainloader, model, view=-1)
            logger.info('-Centroids ready. [Loss: {:.5f}/ Time: {}]\n'.format(kmloss, get_datetime(int(t.time())-int(t1))))
            
            classifier = initialize_classifier(args)
            classifier.module.weight.data = centroids.unsqueeze(-1).unsqueeze(-1)
            freeze_all(classifier)
            
            acc_new, res_new = evaluate(args, logger, testloader, classifier, model)
            acc_list_new.append(acc_new)
            res_list_new.append(res_new)
    else:
        acc_new, res_new = evaluate(args, logger, testloader, classifier, model)
        acc_list_new.append(acc_new)
        res_list_new.append(res_new)

    logger.info('Average overall pixel accuracy [NEW] : {} +/- {}.'.format(round(np.mean(acc_list_new), 2), np.std(acc_list_new)))
    logger.info('Average mIoU [NEW] : {:.3f} +/- {:.3f}. '.format(np.mean([res['mean_iou'] for res in res_list_new]), 
                                                                  np.std([res['mean_iou'] for res in res_list_new])))
    logger.info('Experiment done. [{}]\n'.format(get_datetime(int(t.time())-int(t_start))))
    
if __name__=='__main__':
    args = parse_arguments()

    # Setup the path to save.
    if not args.pretrain:
        args.save_root += '/scratch'
    if args.augment:
        args.save_root += '/augmented/res1={}_res2={}/jitter={}_blur={}_grey={}'.format(args.res1, args.res2, args.jitter, args.blur, args.grey)
    if args.equiv:
        args.save_root += '/equiv/h_flip={}_v_flip={}_crop={}/min_scale\={}'.format(args.h_flip, args.v_flip, args.random_crop, args.min_scale)
    if args.nonparametric:
        args.save_root += '/nonparam'
        
    args.save_model_path = os.path.join(args.save_root, args.comment, 'K_train={}_{}'.format(args.K_train, args.metric_train))
    args.save_eval_path  = os.path.join(args.save_model_path, 'K_test={}_{}'.format(args.K_test, args.metric_test))
    
    if not os.path.exists(args.save_eval_path):
        os.makedirs(args.save_eval_path)

    # Setup logger.
    logger = set_logger(os.path.join(args.save_eval_path, 'train.log'))
    
    # Start.
    main(args, logger)


 

