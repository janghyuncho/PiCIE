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
    parser.add_argument('--repeats', type=int, default=0)  

    # Train. 
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--res', type=int, default=320, help='Input size.')
    parser.add_argument('--res1', type=int, default=320, help='Input size scale from.')
    parser.add_argument('--res2', type=int, default=640, help='Input size scale to.')
    parser.add_argument('--batch_size_cluster', type=int, default=256)
    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optim_type', type=str, default='Adam')
    parser.add_argument('--num_init_batches', type=int, default=30)
    parser.add_argument('--num_batches', type=int, default=30)
    parser.add_argument('--kmeans_n_iter', type=int, default=30)
    parser.add_argument('--in_dim', type=int, default=128)
    parser.add_argument('--X', type=int, default=80)

    # Loss. 
    parser.add_argument('--metric_train', type=str, default='cosine')   
    parser.add_argument('--metric_test', type=str, default='cosine')
    parser.add_argument('--K_train', type=int, default=27) # COCO Stuff-15 / COCO Thing-12 / COCO All-27
    parser.add_argument('--K_test', type=int, default=27) 
    parser.add_argument('--no_balance', action='store_true', default=False)
    parser.add_argument('--mse', action='store_true', default=False)

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
    parser.add_argument('--val_type', type=str, default='train')
    parser.add_argument('--version', type=int, default=7)
    parser.add_argument('--fullcoco', action='store_true', default=False)

    # Eval-only
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--eval_path', type=str)

    return parser.parse_args()





def train(args, logger, dataloader, model, classifier1, classifier2, criterion1, criterion2, optimizer, epoch):
    losses = AverageMeter()
    losses_mse = AverageMeter()
    losses_cet = AverageMeter()
    losses_cet_across = AverageMeter()
    losses_cet_within = AverageMeter()

    # switch to train mode
    model.train()
    if args.mse:
        criterion_mse = torch.nn.MSELoss().cuda()

    classifier1.eval()
    classifier2.eval()
    for i, (indice, input1, input2, label1, label2) in enumerate(dataloader):
        input1 = eqv_transform_if_needed(args, dataloader, indice, input1.cuda(non_blocking=True))
        label1 = label1.cuda(non_blocking=True)
        featmap1 = model(input1)
        
        input2 = input2.cuda(non_blocking=True)
        label2 = label2.cuda(non_blocking=True)
        featmap2 = eqv_transform_if_needed(args, dataloader, indice, model(input2))

        B, C, _ = featmap1.size()[:3]
        if i == 0:
            logger.info('Batch input size   : {}'.format(list(input1.shape)))
            logger.info('Batch label size   : {}'.format(list(label1.shape)))
            logger.info('Batch feature size : {}\n'.format(list(featmap1.shape)))
        
        if args.metric_train == 'cosine':
            featmap1 = F.normalize(featmap1, dim=1, p=2)
            featmap2 = F.normalize(featmap2, dim=1, p=2)

        featmap12_processed, label12_processed = featmap1, label2.flatten()
        featmap21_processed, label21_processed = featmap2, label1.flatten()

        # Cross-view loss
        output12 = feature_flatten(classifier2(featmap12_processed)) # NOTE: classifier2 is coupled with label2
        output21 = feature_flatten(classifier1(featmap21_processed)) # NOTE: classifier1 is coupled with label1
        
        loss12  = criterion2(output12, label12_processed)
        loss21  = criterion1(output21, label21_processed)  

        loss_across = (loss12 + loss21) / 2.
        losses_cet_across.update(loss_across.item(), B)

        featmap11_processed, label11_processed = featmap1, label1.flatten()
        featmap22_processed, label22_processed = featmap2, label2.flatten()
        
        # Within-view loss
        output11 = feature_flatten(classifier1(featmap11_processed)) # NOTE: classifier1 is coupled with label1
        output22 = feature_flatten(classifier2(featmap22_processed)) # NOTE: classifier2 is coupled with label2

        loss11 = criterion1(output11, label11_processed)
        loss22 = criterion2(output22, label22_processed)

        loss_within = (loss11 + loss22) / 2. 
        losses_cet_within.update(loss_within.item(), B)
        loss = (loss_across + loss_within) / 2.
        
        losses_cet.update(loss.item(), B)
        
        if args.mse:
            loss_mse = criterion_mse(featmap1, featmap2)
            losses_mse.update(loss_mse.item(), B)

            loss = (loss + loss_mse) / 2. 
        
        # record loss
        losses.update(loss.item(), B)

        # compute gradient and do step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i % 200) == 0:
            logger.info('{0} / {1}\t'.format(i, len(dataloader)))

    return losses.avg, losses_cet.avg, losses_cet_within.avg, losses_cet_across.avg, losses_mse.avg




def main(args, logger):
    logger.info(args)

    # Use random seed.
    fix_seed_for_reproducability(args.seed)

    # Start time.
    t_start = t.time()

    # Get model and optimizer.
    model, optimizer, classifier1 = get_model_and_optimizer(args, logger)

    # New trainset inside for-loop.
    inv_list, eqv_list = get_transform_params(args)
    trainset = TrainCOCO(args.data_root, res1=args.res1, res2=args.res2,\
                        split='train', mode='compute', labeldir='', inv_list=inv_list, eqv_list=eqv_list, \
                        thing=args.thing, stuff=args.stuff, scale=(args.min_scale, 1)) # NOTE: For now, max_scale = 1.  
    trainloader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=args.batch_size_cluster,
                                                shuffle=False, 
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                collate_fn=collate_train,
                                                worker_init_fn=worker_init_fn(args.seed))
    
    testset    = EvalCOCO(args.data_root, res=args.res, split='val', mode='test', stuff=args.stuff, thing=args.thing)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size_test,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             pin_memory=True,
                                             collate_fn=collate_eval,
                                             worker_init_fn=worker_init_fn(args.seed))
    
    # Before train.
    _, _ = evaluate(args, logger, testloader, classifier1, model)
    
    # Train start.
    for epoch in range(args.start_epoch, args.num_epoch):
        # Assign probs. 
        trainloader.dataset.mode = 'compute'
        trainloader.dataset.reshuffle()

        logger.info('\n============================= [Epoch {}] =============================\n'.format(epoch))
        logger.info('Start computing centroids.')
        t1 = t.time()
        centroids1, kmloss1 = run_mini_batch_kmeans(args, logger, trainloader, model, view=1)
        centroids2, kmloss2 = run_mini_batch_kmeans(args, logger, trainloader, model, view=2)
        logger.info('-Centroids ready. [Loss: {:.5f}| {:.5f}/ Time: {}]\n'.format(kmloss1, kmloss2, get_datetime(int(t.time())-int(t1))))
        
        # Compute cluster assignment. 
        t2 = t.time()
        weight1 = compute_labels(args, logger, trainloader, model, centroids1, view=1)
        weight2 = compute_labels(args, logger, trainloader, model, centroids2, view=2)
        logger.info('-Cluster labels ready. [{}]\n'.format(get_datetime(int(t.time())-int(t2)))) 
        
        # Criterion.
        if not args.no_balance:
            criterion1 = torch.nn.CrossEntropyLoss(weight=weight1).cuda()
            criterion2 = torch.nn.CrossEntropyLoss(weight=weight2).cuda()
        else:
            criterion1 = torch.nn.CrossEntropyLoss().cuda()
            criterion2 = torch.nn.CrossEntropyLoss().cuda()

        # Setup nonparametric classifier.
        classifier1 = initialize_classifier(args)
        classifier2 = initialize_classifier(args)
        classifier1.module.weight.data = centroids1.unsqueeze(-1).unsqueeze(-1)
        classifier2.module.weight.data = centroids2.unsqueeze(-1).unsqueeze(-1)
        freeze_all(classifier1)
        freeze_all(classifier2)

        # Delete since no longer needed. 
        del centroids1 
        del centroids2

        # Set-up train loader.
        trainset.mode  = 'train'
        trainset.labeldir = args.save_model_path
        trainloader_loop  = torch.utils.data.DataLoader(trainset, 
                                                        batch_size=args.batch_size_train, 
                                                        shuffle=True,
                                                        num_workers=args.num_workers,
                                                        pin_memory=True,
                                                        collate_fn=collate_train,
                                                        worker_init_fn=worker_init_fn(args.seed))

        logger.info('Start training ...')
        train_loss, train_cet, cet_within, cet_across, train_mse = train(args, logger, trainloader_loop, model, classifier1, classifier2, criterion1, criterion2, optimizer, epoch) 
        acc1, res1 = evaluate(args, logger, testloader, classifier1, model)
        acc2, res2 = evaluate(args, logger, testloader, classifier2, model)
        
        logger.info('============== Epoch [{}] =============='.format(epoch))
        logger.info('  Time: [{}]'.format(get_datetime(int(t.time())-int(t1))))
        logger.info('  K-Means loss   : {:.5f} | {:.5f}'.format(kmloss1, kmloss2))
        logger.info('  Training Total Loss  : {:.5f}'.format(train_loss))
        logger.info('  Training CE Loss (Total | Within | Across) : {:.5f} | {:.5f} | {:.5f}'.format(train_cet, cet_within, cet_across))
        logger.info('  Training MSE Loss (Total) : {:.5f}'.format(train_mse))
        logger.info('  [View 1] ACC: {:.4f} | mIoU: {:.4f}'.format(acc1, res1['mean_iou']))
        logger.info('  [View 2] ACC: {:.4f} | mIoU: {:.4f}'.format(acc2, res2['mean_iou']))
        logger.info('========================================\n')
        

        torch.save({'epoch': epoch+1, 
                    'args' : args,
                    'state_dict': model.state_dict(),
                    'classifier1_state_dict' : classifier1.state_dict(),
                    'classifier2_state_dict' : classifier2.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    },
                    os.path.join(args.save_model_path, 'checkpoint_{}.pth.tar'.format(epoch)))
        
        torch.save({'epoch': epoch+1, 
                    'args' : args,
                    'state_dict': model.state_dict(),
                    'classifier1_state_dict' : classifier1.state_dict(),
                    'classifier2_state_dict' : classifier2.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    },
                    os.path.join(args.save_model_path, 'checkpoint.pth.tar'))
    
    # Evaluate.
    trainset    = EvalCOCO(args.data_root, res=args.res, split=args.val_type, mode='test', label=False) 
    trainloader = torch.utils.data.DataLoader(trainset, 
                                                batch_size=args.batch_size_cluster,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True,
                                                collate_fn=collate_train,
                                                worker_init_fn=worker_init_fn(args.seed))

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
            centroids1, kmloss1 = run_mini_batch_kmeans(args, logger, trainloader, model, view=-1)
            logger.info('-Centroids ready. [Loss: {:.5f}/ Time: {}]\n'.format(kmloss1, get_datetime(int(t.time())-int(t1))))
            
            classifier1 = initialize_classifier(args)
            classifier1.module.weight.data = centroids1.unsqueeze(-1).unsqueeze(-1)
            freeze_all(classifier1)
            
            acc_new, res_new = evaluate(args, logger, testloader, classifier1, model)
            acc_list_new.append(acc_new)
            res_list_new.append(res_new)
    else:
        acc_new, res_new = evaluate(args, logger, testloader, classifier1, model)
        acc_list_new.append(acc_new)
        res_list_new.append(res_new)

    logger.info('Average overall pixel accuracy [NEW] : {:.3f} +/- {:.3f}.'.format(np.mean(acc_list_new), np.std(acc_list_new)))
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
    if args.no_balance:
        args.save_root += '/no_balance'
    if args.mse:
        args.save_root += '/mse'

    args.save_model_path = os.path.join(args.save_root, args.comment, 'K_train={}_{}'.format(args.K_train, args.metric_train))
    args.save_eval_path  = os.path.join(args.save_model_path, 'K_test={}_{}'.format(args.K_test, args.metric_test))
    
    if not os.path.exists(args.save_eval_path):
        os.makedirs(args.save_eval_path)

    # Setup logger.
    logger = set_logger(os.path.join(args.save_eval_path, 'train.log'))
    
    # Start.
    main(args, logger)
