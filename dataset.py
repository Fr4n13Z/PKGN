import pickle

import torch

from utils import pad_input_tokens
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch_geometric.utils import to_undirected


class GATDataset(Dataset):
    def __init__(self, args):
        super(GATDataset, self).__init__()
        self.args = args
        self.cate_list = ['[PRON]', '[PREP]', '[AUXV]', '[INTE]', '[AFFE]', '[SOCI]', '[COGP]', '[PERC]', '[BIOL]',
                          '[DRIV]', '[RELA]', '[INFO]', '[WORK]', '[LEIS]', '[HOME]', '[MONE]', '[RELI]', '[DEAT]']
        self.cate_list = [i.lower() for i in self.cate_list]
        self.label_lookup = {'E': 1, 'I': 0, 'S': 1, 'N': 0, 'T': 1, 'F': 0, 'J': 1, 'P': 0}
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path, local_files_only=True)
        self.tokenizer.add_tokens(self.cate_list, special_tokens=True)
        if args.load_saved_data is False:
            with open(args.data_path, 'rb') as f:
                data_dict = pickle.load(f)
                self.features = data_dict['features']
                self.labels = data_dict['labels']
                self.edges = data_dict['edges']
                f.close()
            self.tokenize_data()
        else:
            self.load_datas(args.saved_data)

    def tokenize_data(self):
        for i in range(len(self.features)):
            feature_i = []
            attention_mask_i = []
            posts = self.features[i]
            posts_mask = [0] * self.args.cate_length
            for j, post in enumerate(posts[:self.args.max_node]):
                if j >= 18:
                    posts_mask.append(1)
                tokens = self.tokenizer.tokenize(post)
                context, attention_mask = pad_input_tokens(tokens, self.args.max_length)
                post_tokens_id = self.tokenizer.convert_tokens_to_ids(context)
                feature_i.append(post_tokens_id)
                attention_mask_i.append(attention_mask)
            if len(feature_i) < self.args.max_node:
                pad_length = (self.args.max_node - len(feature_i))
                context, attention_mask = pad_input_tokens(['[PAD]'] * (self.args.max_length - 2), self.args.max_length)
                feature_i += [self.tokenizer.convert_tokens_to_ids(context)] * pad_length
                attention_mask_i += [attention_mask] * pad_length
                posts_mask += [0] * pad_length
            self.features[i] = {
                "input_ids": feature_i,
                "attention_mask_ids": attention_mask_i,
                "posts_mask": posts_mask
            }
            self.labels[i] = [self.label_lookup[list(self.labels[i])[0]],  # EI
                              self.label_lookup[list(self.labels[i])[1]],  # SN
                              self.label_lookup[list(self.labels[i])[2]],  # TF
                              self.label_lookup[list(self.labels[i])[3]]]  # JP

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return torch.Tensor(self.features[item]['input_ids']).long(), \
               torch.Tensor(self.features[item]['posts_mask']), \
               torch.Tensor(self.features[item]['attention_mask_ids']).long(),\
               to_undirected(torch.Tensor(self.edges[item]).long().t()),\
               torch.Tensor(self.labels[item]).long()

    def dump_datas(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump({
                "features": self.features,
                "labels": self.labels,
                "edges": self.edges,
            }, f)
            f.close()

    def load_datas(self, file_path):
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
            f.close()
        self.features = data_dict['features']
        self.labels = data_dict['labels']
        self.edges = data_dict['edges']

    def cate_ablation_study(self, idx: int):
        for i in range(self.__len__()):
            self.features[i]["input_ids"][idx] = [0 for _ in range(self.args.max_length)]
            self.features[i]["attention_mask_ids"][idx] = [0 for _ in range(self.args.max_length)]
            self.features[i]["posts_mask"][idx] = 0
        for i in range(self.__len__()):
            edges = self.edges[i]
            for j in self.edges[i]:
                if int(idx) in j:
                    edges.remove(j)
            self.edges[i] = edges
