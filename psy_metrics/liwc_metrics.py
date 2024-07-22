# -*- coding: utf-8 -*-
import gc
import math
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_liwc(file_path):
    categories_id = {2: '[PRON]', 3:'[PREP]', 12: '[AUXV]', 23: '[INTE]', 30: '[AFFE]', 40: '[SOCI]', 50: '[COGP]', 60: '[PERC]',
                     70: '[BIOL]', 80: '[DRIV]', 100: '[RELA]', 120: '[INFO]', 110: '[WORK]', 111: '[LEIS]',
                     112: '[HOME]', 113: '[MONE]', 114: '[RELI]', 115: '[DEAT]'}
    CAT_DELIM = "%"
    with open(file_path, 'r') as f:
        categories_section = False
        words_to_categories = {}
        id_to_cat = {}
        for line in f:
            line = line.strip()
            if line == CAT_DELIM:
                categories_section = not categories_section
                continue
            if categories_section:
                try:
                    i, category = line.split('\t')
                    category = category.split()[0]
                    id_to_cat[int(i)] = category
                except:
                    pass
            else:
                w, categories = line.split('\t')[0], line.split('\t')[1:]
                if "(" in w and ")" in w:
                    w = w.replace("(", "").replace(")", "")
                words_categories = []
                for i in categories:
                    try:
                        words_categories.append(categories_id[int(i)])
                    except:
                        continue
                if words_categories:
                    words_to_categories[w] = words_categories
    return words_to_categories, categories_id


def words2categories(words_to_categories, word):
    word_categories = []
    try:
        word_categories = words_to_categories[word]
        return word_categories
    except:
        for i in range(len(word)):
            try:
                word_categories = words_to_categories[word[:i + 1] + '*']
            except:
                continue
    return word_categories


def linguistic_score(score_list, post_len, aggr_type='sum'):
    sum_score = 0.
    c_i = 0
    res_list = []
    for i in score_list:
        score = math.log2(1. + (i / post_len))
        res_list.append(score)
        if score > 0:
            c_i += 1
        sum_score += score
    if aggr_type == 'sum':
        if c_i == 0:
            return res_list, 0.0
        else:
            return res_list, sum_score
    else:
        if c_i == 0:
            return res_list, 0.0
        else:
            return res_list, sum_score / c_i


def post_psycho_metrics(tokenizer, words_to_categories, post, drop_words, id2idx, aggr_type):
    score_list = [0 for _ in range(18)]
    words = tokenizer.tokenize(post)
    if len(words) == 0:
        return 0.0
    one_word = []
    length = 0
    for j, word in enumerate(words):
        one_word.append(word)
        try:
            next_word = words[j+1]
            if '##' in next_word:
                continue
            else:
                length += 1
                right_words = ''.join(one_word).replace('##', '') if len(one_word) > 1 else one_word[0]
                word_categories = words2categories(words_to_categories, right_words)
                if word_categories:
                    for k in word_categories:
                        score_list[id2idx[k]] += 1
                one_word = []
        except:
            right_words = ''.join(one_word).replace('##', '') if len(one_word) > 1 else one_word[0]
            word_categories = words2categories(words_to_categories, right_words)
            if word_categories:
                for k in word_categories:
                    score_list[id2idx[k]] += 1
            else:
                drop_words.append(right_words)
            one_word = []
    if length <= 5:
        return [], 0.0
    res_list, score = linguistic_score(score_list, length, aggr_type)
    # print(res_list, score)
    return res_list, score


def user_post_metric(bert_path: str, liwc_path: str, user_posts: pd.DataFrame, aggr_type, output_file=None):
    print("user post metric:")
    score_list = []
    cate_list = ['[PRON]', '[PREP]', '[AUXV]', '[INTE]', '[AFFE]', '[SOCI]', '[COGP]', '[PERC]', '[BIOL]', '[DRIV]',
                 '[RELA]', '[INFO]', '[WORK]', '[LEIS]', '[HOME]', '[MONE]', '[RELI]', '[DEAT]']
    id2idx = {cate_list[i]: i for i in range(len(cate_list))}
    tokenizer = AutoTokenizer.from_pretrained(bert_path, local_files_only=True)
    words_to_categories, categories_id = parse_liwc(liwc_path)
    for i in tqdm(range(user_posts.shape[0])):
        drop_words = []
        post = user_posts.iloc[i, 1]
        if post != '':
            _, score = post_psycho_metrics(tokenizer, words_to_categories, post, drop_words, id2idx, aggr_type)
            score_list.append(score)
        else:
            score_list.append(0.0)
    user_posts['psy_metric'] = score_list
    user_posts = user_posts.drop(index=user_posts[user_posts['psy_metric'] == 1.0].index, axis=1)
    user_posts = user_posts.drop(index=user_posts[user_posts['psy_metric'] == 2.0].index, axis=1)
    user_posts = user_posts.drop(index=user_posts[user_posts['psy_metric'] == 3.0].index, axis=1)
    user_posts = user_posts.drop(index=user_posts[user_posts['psy_metric'] == 0.0].index, axis=1)
    if output_file is not None:
        user_posts.to_csv(output_file, index=False, index_label=False)
    return user_posts


def neighbor_post_metric(bert_path: str, liwc_path: str, neighbor_posts: pd.DataFrame, aggr_type: str, output_file=None):
    print("neighbor post metric:")
    score_list = []
    cate_list = ['[PRON]', '[PREP]', '[AUXV]', '[INTE]', '[AFFE]', '[SOCI]', '[COGP]', '[PERC]', '[BIOL]', '[DRIV]',
                 '[RELA]', '[INFO]', '[WORK]', '[LEIS]', '[HOME]', '[MONE]', '[RELI]', '[DEAT]']
    id2idx = {cate_list[i]: i for i in range(len(cate_list))}
    tokenizer = AutoTokenizer.from_pretrained(bert_path, local_files_only=True)
    words_to_categories, categories_id = parse_liwc(liwc_path)
    for i in tqdm(range(neighbor_posts.shape[0])):
        drop_words = []
        post = neighbor_posts.iloc[i, 1]
        if post != '':
            _, score = post_psycho_metrics(tokenizer, words_to_categories, post, drop_words, id2idx, aggr_type)
            score_list.append(score)
        else:
            score_list.append(0.0)
    neighbor_posts['psy_metric'] = score_list
    neighbor_posts = neighbor_posts.drop(index=neighbor_posts[neighbor_posts['psy_metric'] == 1.0].index, axis=1)
    neighbor_posts = neighbor_posts.drop(index=neighbor_posts[neighbor_posts['psy_metric'] == 2.0].index, axis=1)
    neighbor_posts = neighbor_posts.drop(index=neighbor_posts[neighbor_posts['psy_metric'] == 3.0].index, axis=1)
    neighbor_posts = neighbor_posts.drop(index=neighbor_posts[neighbor_posts['psy_metric'] == 0.0].index, axis=1)
    neighbor_data = pd.DataFrame({
        "ID": neighbor_posts['user_id_str'].tolist(),
        "post": neighbor_posts['full_text'].tolist(),
        'users': neighbor_posts['users'].tolist(),
        'psy_metric': neighbor_posts['psy_metric'].tolist(),
    })
    if output_file is not None:
        neighbor_data.to_csv(output_file, index=False, index_label=False)
    return neighbor_data


def metrics_example():
    cate_list = ['[PRON]', '[PREP]', '[AUXV]', '[INTE]', '[AFFE]', '[SOCI]', '[COGP]', '[PERC]', '[BIOL]', '[DRIV]',
                 '[RELA]', '[INFO]', '[WORK]', '[LEIS]', '[HOME]', '[MONE]', '[RELI]', '[DEAT]']
    id2idx = {cate_list[i]: i for i in range(len(cate_list))}
    tokenizer = AutoTokenizer.from_pretrained("../../Pretrained_models/bert-base-uncased",
                                              local_files_only=True)
    words_to_categories, categories_id = parse_liwc('./LIWC/LIWC2015_English.dic')
    post = "Hello, World!"
    drop_words = []
    post_psycho_metrics(tokenizer, words_to_categories, post, drop_words, id2idx, 'sum')