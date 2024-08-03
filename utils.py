# -*- coding: utf-8 -*-
import logging
import os
from typing import Any

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from sklearn.metrics import f1_score

from tqdm import tqdm
from sklearn.model_selection import KFold


def pad_input_tokens(tokens, max_length):
    context = ["[CLS]"] + tokens[:max_length - 2] + ["[SEP]"]
    attention_mask = [1] * len(context)
    while len(context) < max_length:
        context.append("[PAD]")
        attention_mask.append(0)
    # attention_mask[-1] = 0
    assert len(context) == len(attention_mask)
    return context, attention_mask


def index2graph(edge_index, max_node=500):
    node_num = max_node
    adj = np.eye(node_num)
    for i, j in edge_index:
        if i < node_num and j < node_num:
            adj[i, j] = 1
    return adj


def dim_acc_f1(predictions, labels):
    acc = (predictions == labels).float().mean()
    f1 = f1_score(y_pred=predictions.cpu().numpy(), y_true=labels.cpu().numpy(), average='macro')
    return acc, f1


def get_evaluation(predictions, labels, logger):
    predictions = torch.cat(predictions, dim=0)
    labels = torch.cat(labels, dim=0)
    IE_acc, IE_f1 = dim_acc_f1(predictions[:, 0], labels[:, 0])
    NS_acc, NS_f1 = dim_acc_f1(predictions[:, 1], labels[:, 1])
    FT_acc, FT_f1 = dim_acc_f1(predictions[:, 2], labels[:, 2])
    PJ_acc, PJ_f1 = dim_acc_f1(predictions[:, 3], labels[:, 3])
    total_acc, total_f1 = (IE_acc + NS_acc + FT_acc + PJ_acc) / 4, (IE_f1 + NS_f1 + FT_f1 + PJ_f1) / 4
    logger.info('IE_acc = {:,.4f}, IE_f1 = {:,.4f}'.format(IE_acc, IE_f1))
    logger.info('NS_acc = {:,.4f}, NS_f1 = {:,.4f}'.format(NS_acc, NS_f1))
    logger.info('FT_acc = {:,.4f}, FT_f1 = {:,.4f}'.format(FT_acc, FT_f1))
    logger.info('PJ_acc = {:,.4f}, PJ_f1 = {:,.4f}'.format(PJ_acc, PJ_f1))
    logger.info('total_acc = {:,.4f}, total_f1 = {:,.4f}'.format(total_acc, total_f1))
    return total_f1


def train_process(args: Any, model, optimizer, train_dataloader, test_dataloader, device):
    logger = logging.getLogger(args.main_name)
    best_test_f1 = 0.
    best_epoch = 0
    epoch_steps = len(train_dataloader)
    for epoch in range(args.epoch):
        model.train()
        preds = []
        annos = []
        train_len = 0
        train_loss = 0.
        train_acc = 0.
        for step, datas in tqdm(enumerate(train_dataloader)):
            posts_ids, posts_mask, attn_mask, edges, labels = [data.to(device) for data in datas]
            outputs = model(
                posts_ids=posts_ids,
                posts_mask=posts_mask,
                attention_mask=attn_mask,
                edges=edges,
                labels=labels
            )
            preds.append(outputs['pred'].cpu())
            annos.append(labels.cpu())
            labels_size = labels.numel()
            loss = outputs['loss'].sum() / labels_size
            acc = outputs['acc'].sum() / labels_size
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or step + 1 == epoch_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            train_len += labels_size
            train_loss += labels_size * loss.item() * args.gradient_accumulation_steps
            train_acc += labels_size * acc.item()
        logger.info('epoch = {}, train_result:'.format(epoch))
        get_evaluation(preds, annos, logger)
        with torch.no_grad():
            # torch.save(model, os.path.join('result', args.output_dir, 'MBTI_Result.pt'))
            torch.save(model.state_dict(), str(os.path.join(args.output_dir, args.save_path)))
            test_loss, test_acc, preds, annos = test_process(args, model, test_dataloader, device)
            final_test_f1 = get_evaluation(preds, annos, logger)
            logger.info('epoch = {}, test_result:'.format(epoch))
            if final_test_f1 > best_test_f1:
                best_test_f1, best_epoch = final_test_f1, epoch
                torch.save(model.state_dict(), str(os.path.join(args.output_dir, args.save_path)))
        logger.info(
            'best_epoch = {}, final_test_f1 = {:,.4f}, |||| best_test_f1 = {:,.4f}, \n'.format(
                best_epoch, final_test_f1, best_test_f1))


def test_process(args: Any, model, test_dataloader, device):
    model.eval()
    test_len = 0.
    test_acc = 0.
    test_loss = 0.
    preds = []
    annos = []
    for step, datas in tqdm(enumerate(test_dataloader)):
        posts_ids, posts_mask, attn_mask, edges, labels = [data.to(device) for data in datas]
        outputs = model(
            posts_ids=posts_ids,
            posts_mask=posts_mask,
            attention_mask=attn_mask,
            edges=edges,
            labels=labels
        )
        preds.append(outputs['pred'].cpu())
        annos.append(labels.cpu())

        labels_size = labels.numel()
        loss = outputs['loss'].sum() / labels_size
        acc = outputs['acc'].sum() / labels_size

        test_len += labels_size
        test_loss += labels_size * loss.item()
        test_acc += labels_size * acc.item()

    return test_loss / test_len, test_acc / test_len, preds, annos
