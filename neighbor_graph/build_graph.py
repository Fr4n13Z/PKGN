# -*- coding: utf-8 -*-
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer
from Data.data_utils import neighbor_post_select, add_labels_into_neighbor_dict, random_neighbor_post_select
from psy_metrics.liwc_metrics import words2categories, parse_liwc


cate_list = ['[PRON]', '[PREP]', '[AUXV]', '[INTE]', '[AFFE]', '[SOCI]', '[COGP]', '[PERC]', '[BIOL]',
                 '[DRIV]', '[RELA]', '[INFO]', '[WORK]', '[LEIS]', '[HOME]', '[MONE]', '[RELI]', '[DEAT]']


def get_categories(tokenizer, words_to_categories, post, drop_words, id2idx):
    categories = []
    words = tokenizer.tokenize(post)
    if len(words) == 0:
        return []
    one_word = []
    for j, word in enumerate(words):
        one_word.append(word)
        try:
            next_word = words[j+1]
            if '##' in next_word:
                continue
            else:
                right_words = ''.join(one_word).replace('##', '') if len(one_word) > 1 else one_word[0]
                if right_words not in drop_words:
                    word_categories = words2categories(words_to_categories, right_words)
                    if word_categories:
                        categories += id2idx[word_categories]
                    else:
                        drop_words.append(right_words)
                one_word = []
        except:
            right_words = ''.join(one_word).replace('##', '') if len(one_word) > 1 else one_word[0]
            if right_words not in drop_words:
                word_categories = words2categories(words_to_categories, right_words)
                if word_categories:
                    categories += word_categories
                else:
                    drop_words.append(right_words)
            one_word = []
    return list(set(categories))


def single_neighbor_graph(posts_list: list, tokenizer, words_to_categories, id2idx, max_node=200):
    node_features = list(id2idx.keys())
    node_idx = len(node_features) - 1
    edges = []
    for posts in posts_list:
        if len(posts) > 0 and node_idx < max_node:
            for post_i in posts:
                node_idx += 1
                post = post_i
                node_features.append(post)
                drop_words = []
                categories = get_categories(tokenizer, words_to_categories, post, drop_words, id2idx)
                for item in categories:
                    cate_idx = id2idx[item]
                    edges.append([cate_idx, node_idx])
                    edges.append([node_idx, cate_idx])
    return node_features, edges


def get_categories_example():
    cate_list = ['[PRON]', '[PREP]', '[AUXV]', '[INTE]', '[AFFE]', '[SOCI]', '[COGP]', '[PERC]', '[BIOL]',
                 '[DRIV]', '[RELA]', '[INFO]', '[WORK]', '[LEIS]', '[HOME]', '[MONE]', '[RELI]', '[DEAT]']
    id2idx = {cate_list[i]: i for i in range(len(cate_list))}
    tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased",
                                              local_files_only=True)
    words_to_categories, categories_id = parse_liwc('../psy_metrics/LIWC/LIWC2015_English.dic')
    post = "if you dont have anyone to advise you, woe unto you, but if you have someone to advise you and " \
           "you refuse the advise in toto, then a bigger woe unto you!"
    drop_words = []
    print(get_categories(tokenizer, words_to_categories, post, drop_words, id2idx))


def build_neighbor_dict_example():
    neighbor_post_select("../bert-base-uncased",
                         "../psy_metrics/LIWC/LIWC2015_English.dic",
                         "../Data/target_person/friend_selection.csv",
                         '../Data/neighbor_person/neighbor_posts.csv',
                         './neighbor_post_select_avg.pkl',
                         "avg")
    add_labels_into_neighbor_dict('../Data/labels/label_choose.csv',
                                  './neighbor_post_select_avg.pkl',
                                  './neighbor_post_select_avg.pkl')


def build_graph_example():
    id2idx = {cate_list[i]: i for i in range(len(cate_list))}
    tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased", local_files_only=True)
    words_to_categories, categories_id = parse_liwc('../psy_metrics/LIWC/LIWC2015_English.dic')
    with open("../neighbor_graph/neighbor_post_select_avg.pkl", 'rb') as f:
        post_dict = pickle.load(f)
        f.close()
    graph_features = []
    edge_list = []
    labels = []
    for i in tqdm(range(len(post_dict['ID']))):
        node_features, node_edges = single_neighbor_graph(post_dict['posts'][i], tokenizer, words_to_categories, id2idx, 220)
        graph_features.append(node_features)
        edge_list.append(node_edges)
        labels.append(post_dict['MBTI'][i])
    with open("../neighbor_graph/neighbor_graph_avg.pkl", 'wb') as f:
        pickle.dump({"features": graph_features,
                     "edges": edge_list,
                     "labels": labels,
                     }, f)
        f.close()


def build_random_neighbor_dict_example():
    random_neighbor_post_select("../Data/target_person/friend_selection.csv",
                                '../Data/neighbor_person/neighbor_posts.csv',
                                './random_neighbor_post_select.pkl')
    add_labels_into_neighbor_dict('../Data/labels/label_choose.csv',
                                  './random_neighbor_post_select.pkl',
                                  './random_neighbor_post_select.pkl')


def build_random_graph_example():
    id2idx = {cate_list[i]: i for i in range(len(cate_list))}
    tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased", local_files_only=True)
    words_to_categories, categories_id = parse_liwc('../psy_metrics/LIWC/LIWC2015_English.dic')
    with open("../neighbor_graph/random_neighbor_post_select.pkl", 'rb') as f:
        post_dict = pickle.load(f)
        f.close()
    graph_features = []
    edge_list = []
    labels = []
    for i in tqdm(range(len(post_dict['ID']))):
        node_features, node_edges = single_neighbor_graph(post_dict['posts'][i], tokenizer, words_to_categories, id2idx, 220)
        graph_features.append(node_features)
        edge_list.append(node_edges)
        labels.append(post_dict['MBTI'][i])
    with open("../neighbor_graph/random_neighbor_graph.pkl", 'wb') as f:
        pickle.dump({"features": graph_features,
                     "edges": edge_list,
                     "labels": labels,
                     }, f)
        f.close()


if __name__ == '__main__':
    build_neighbor_dict_example()
    build_graph_example()