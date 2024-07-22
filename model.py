# -*- coding: utf-8 -*-
import logging
import os
from typing import Any

import torch
import torch.nn.functional as F

from sklearn.metrics import f1_score
from tensorboardX import SummaryWriter
from torch import nn
from torch_geometric.nn import GATv2Conv
from transformers import BertConfig, BertModel


class BERT(nn.Module):
    def __init__(self, args: Any, tokenizer):
        super(BERT, self).__init__()
        self.tokenizer = tokenizer
        self.args = args
        self.bert_config = BertConfig.from_pretrained(args.bert_path, local_files_only=True)
        self.bert_config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(args.bert_path, config=self.bert_config, local_files_only=True)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.pad_id = 0

    def forward(self, nodes_ids=None, attention_mask=None):
        node_ids = nodes_ids.view(-1, self.args.max_length)
        attention_mask = attention_mask.view(-1, self.args.max_length)
        cls_token = self.bert(input_ids=node_ids, attention_mask=attention_mask)[0][:, :1]
        last_semantic_reply = cls_token.view(-1, self.args.hidden_size)
        return last_semantic_reply


class PKGN(nn.Module):
    def __init__(self, args: Any, tokenizer):
        super(PKGN, self).__init__()
        self.emb_encoder = BERT(args, tokenizer)
        self.args = args
        self.gat = GATv2Conv(args.hidden_size,
                             args.hidden_size,
                             args.head_num,
                             dropout=args.dropout,
                             negative_slope=args.alpha,
                             concat=False)
        self.classifiers = nn.ModuleList([nn.Linear(args.hidden_size, 2) for _ in range(args.label_num)])

    def forward(self,
                posts_ids,
                posts_mask,
                attention_mask,
                edges,
                labels):
        emb_vector = self.emb_encoder(posts_ids, attention_mask)
        edges = edges.reshape(edges.shape[1], edges.shape[2])
        gat_vector = self.gat(emb_vector, edges)
        last_psy_vector = gat_vector.masked_fill((1-posts_mask)[:, :, None].reshape(self.args.max_node, 1).expand_as(gat_vector).bool(), 0).\
                              sum(dim=-2) / (posts_mask.sum(dim=-1)[:, None].expand(-1, self.args.hidden_size) + 1e-8)
        logits = torch.stack([torch.softmax(classifier(last_psy_vector), dim=-1) for classifier in self.classifiers], dim=1)
        outputs = {'loss': F.cross_entropy(logits.view(-1, 2), labels.view(-1), reduction='sum'),
                   'pred': torch.argmax(logits.view(-1, 2), dim=-1).view(-1, 4)}
        outputs['acc'] = (outputs['pred'] == labels).float().sum()
        return outputs

