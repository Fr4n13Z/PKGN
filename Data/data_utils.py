# -*- coding: utf-8 -*-
import pickle
import random
import re
import unicodedata

import emoji
import numpy as np
import pandas as pd
from tqdm import tqdm

from psy_metrics.liwc_metrics import neighbor_post_metric, user_post_metric


def text_symbol_preprocessing(text: str, lower: bool) -> str:
    text = emoji.demojize(text, language='en')
    if lower is True:
        text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[^\x00-\x7F]+', "", text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'http?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'@[\w]*', '', text)
    text = re.sub(r'#[\w]*', '', text)
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'").replace("：", ":")
    text = text.replace("•", "")
    text = text.replace('rt :', "")
    text = text.replace('(/n)', '')
    text = text.encode('ascii', 'ignore').decode('ascii')
    if text.startswith("'"):
        text = text[1:-1]
    if text.startswith('"'):
        text = text[1:-1]
    return ' '.join(text.split())


def find_emoji(text):
    pattern = re.compile('(\:.*?\:)')
    res = re.findall(pattern, text)
    return res


def find_user(text):
    if pd.isnull(text):
        return text
    else:
        pattern = re.compile(r'@[\w]*')
        res = re.findall(pattern, text)
        res_str = "|||".join(res).replace("@", "")
        return res_str


def user_post_clean(post_file: str):
    print("cleaning the user posts:")
    user_posts = pd.read_csv(post_file, dtype=str)
    user_posts = user_posts[user_posts['lang'] == 'en']
    for i in tqdm(range(0, user_posts.shape[0])):
        post = user_posts.iloc[i, 6]
        post = text_symbol_preprocessing(post, lower=True)
        user_posts.iloc[i, 6] = post
    user_posts = user_posts.drop(index=user_posts[user_posts['post'] == ''].index, axis=1)
    user_posts = user_posts.iloc[:, [1, 6, 7, 8]]
    return user_posts


def user_post_select(bert_path, liwc_path, post_file: str, output_file: str, aggr_type: str):
    post_data = user_post_clean(post_file)
    post_data = user_post_metric(bert_path=bert_path,
                                 liwc_path=liwc_path,
                                 user_posts=post_data,
                                 aggr_type=aggr_type)
    user_ids = list(set(post_data['user_id']))
    print("user post selection:")
    post_nest = []
    length = []
    for i in tqdm(user_ids):
        posts = (post_data.loc[post_data['user_id'] == i, :].nlargest(50, ['psy_metric'], keep='first'))[
            'post'].tolist()
        length.append(len(posts))
        post_nest.append("|||".join(posts))
    final_data = pd.DataFrame({
        "ID": user_ids,
        "posts": post_nest,
        "post_num": length,
    })
    final_data.to_csv(output_file, index=False, index_label=False)
    return output_file


def split_username(text):
    text = text.split("/")[-1]
    return text


def label_reprocess(label_file: str, profile_file: str, output_file: str):
    print("label reprocessing:")
    try:
        label_data = pd.read_excel(label_file)
    except:
        label_data = pd.read_csv(label_file, dtype=str)
    profile_data = pd.read_csv(profile_file, dtype=str)
    label_data = label_data.loc[:, ['name', 'username', 'MBTI']]
    ids = []
    for i in tqdm(range(label_data.shape[0])):
        user_id = profile_data.loc[profile_data['name'] == label_data.iloc[i, 0], 'id'].values
        if len(user_id) > 0:
            user_id = user_id[0]
        else:
            user_id = profile_data.loc[profile_data['screen_name'] == label_data.iloc[i, 1], 'id'].values
            if len(user_id) > 0:
                user_id = user_id[0]
            else:
                user_id = np.nan
        ids.append(user_id)
    label_data['ID'] = ids
    label_data.to_csv(output_file, index=False, index_label=False)
    return label_data


def add_labels_into_post(label_file: str, post_file: str, output_file: str):
    print("adding labels into user posts:")
    label_data = pd.read_csv(label_file, dtype=str)
    post_data = pd.read_csv(post_file, dtype=str)
    label_list = []
    for i in tqdm(range(post_data.shape[0])):
        label = label_data.loc[label_data['ID'] == post_data.iloc[i, 0], 'MBTI'].values
        if len(label) > 0:
            label_list.append(label[0])
        else:
            label_list.append(np.nan)
    post_data['MBTI'] = label_list
    post_data.to_csv(output_file, index=False, index_label=False)
    return post_data


def add_labels_into_neighbor_dict(label_file: str, dict_file: str, output_file: str):
    print("adding labels into neighbor posts:")
    label_data = pd.read_csv(label_file, dtype=str)
    with open(dict_file, 'rb') as f:
        neighbor_dict = pickle.load(f)
        f.close()
    labels = []
    posts = []
    ids = []
    for i in tqdm(range(len(neighbor_dict['user_id']))):
        label = label_data.loc[label_data['ID'] == neighbor_dict['user_id'][i], 'MBTI'].values
        if len(label) > 0:
            posts.append(neighbor_dict['neighbor_posts'][i])
            ids.append(neighbor_dict['user_id'][i])
            labels.append(label[0])
    with open(output_file, 'wb') as f:
        pickle.dump({
            "ID": ids,
            "posts": posts,
            "MBTI": labels,
        }, f)
        f.close()
    return {"ID": ids, "posts": posts, "MBTI": labels,}


def neighbor_select(profile_file: str, output_file: str):
    print("neighbor selection:")
    person_profile = pd.read_csv(profile_file, dtype=str)
    friend_selection = person_profile.loc[:,
                       ['screen_name', 'name', 'id', 'top_friendsNum', 'top_mentionedNum', 'top_postNum']]
    friend_selection = friend_selection.dropna(how='any',
                                               subset=['id', 'top_friendsNum', 'top_mentionedNum', 'top_postNum'])
    friends_list = []
    for i in tqdm(range(friend_selection.shape[0])):
        top_friendsNum = friend_selection.iloc[i, -3].split('|||')
        top_mentionedNum = friend_selection.iloc[i, -2].split('|||')
        top_postNum = friend_selection.iloc[i, -1].split('|||')
        fid_list = list(set(top_mentionedNum))
        if len(fid_list) < 20:
            fid_list = list(set(top_mentionedNum + top_friendsNum))
            if len(fid_list) < 20:
                fid_list = list(set(top_mentionedNum + top_friendsNum + top_postNum))
                if len(fid_list) > 20:
                    fid_list = fid_list[:20]
            else:
                fid_list = fid_list[:20]
        friends_list.append("|||".join(fid_list))
    friend_selection['selection'] = friends_list
    friend_selection.to_csv(output_file, index=False, index_label=False)
    return friend_selection


def get_neighbor_ids(relation_file: str):
    print("get neighbor ids:")
    friend_selection = pd.read_csv(relation_file, dtype=str)
    user_ids = friend_selection['id'].tolist()
    for i in tqdm(range(len(user_ids))):
        top_friendsNum = friend_selection.iloc[i, -4].split('|||')
        top_mentionedNum = friend_selection.iloc[i, -3].split('|||')
        top_postNum = friend_selection.iloc[i, -2].split('|||')
        user_ids += top_friendsNum
        user_ids += top_mentionedNum
        user_ids += top_postNum
    user_ids = list(set(user_ids))
    return user_ids


def neighbor_post_clean(neighbor_post_file: str, output_file=None) -> pd.DataFrame:
    print("cleaning the neighbor user posts:")
    neighbor_posts = pd.read_csv(neighbor_post_file, dtype=str)
    neighbor_posts = neighbor_posts[neighbor_posts['lang'] == 'en']
    for i in tqdm(range(neighbor_posts.shape[0])):
        neighbor_posts.iloc[i, 1] = text_symbol_preprocessing(neighbor_posts.iloc[i, 1], True)
    neighbor_posts = neighbor_posts.dropna(how='any', subset=['full_text'])
    if output_file is not None:
        neighbor_posts.to_csv(output_file, index=False, index_label=False)
    return neighbor_posts


def neighbor_post_select(bert_path, liwc_path, relation_file: str, neighbor_post_file: str, output_file: str, aggr_type: str):
    neighbor_posts = neighbor_post_clean(neighbor_post_file, neighbor_post_file[:-4]+"_clean.csv")
    neighbor_data = neighbor_post_metric(bert_path=bert_path,
                                         liwc_path=liwc_path,
                                         neighbor_posts=neighbor_posts,
                                         aggr_type=aggr_type)
    print("neighbor post selection:")
    friends_relation = pd.read_csv(relation_file, dtype=str)
    friends_post_list = []
    user_id_list = []
    for i in tqdm(range(friends_relation.shape[0])):
        selection = friends_relation.iloc[i, -1]
        selection_list = selection.split('|||')
        friend_post_i = []
        for j in selection_list:
            friend_id = str(j)
            try:
                friend_posts = \
                (neighbor_data.loc[neighbor_data['ID'] == friend_id, :].nlargest(10, ['psy_metric'], keep='first'))[
                    'post'].tolist()
            except:
                friend_posts = None
            friend_post_i.append(friend_posts)
        friends_post_list.append(friend_post_i)
        user_id_list.append(friends_relation.iloc[i, 2])
    selected_dict = {"user_id": user_id_list,
                     "neighbor_posts": friends_post_list,
                     }
    with open(output_file, 'wb') as f:
        pickle.dump(selected_dict, f)
        f.close()
    return selected_dict


def random_neighbor_post_select(relation_file: str, neighbor_post_file: str, output_file: str):
    neighbor_posts = neighbor_post_clean(neighbor_post_file)
    neighbor_data = pd.DataFrame({
        "ID": neighbor_posts['user_id_str'].tolist(),
        "post": neighbor_posts['full_text'].tolist(),
    })
    print("random neighbor post selection:")
    friends_relation = pd.read_csv(relation_file, dtype=str)
    friends_post_list = []
    user_id_list = []
    for i in tqdm(range(friends_relation.shape[0])):
        selection = friends_relation.iloc[i, -1]
        selection_list = selection.split('|||')
        friend_post_i = []
        for j in selection_list:
            try:
                friend_id = str(j)
                friend_posts = neighbor_data.loc[neighbor_data['ID'] == friend_id, 'post'].tolist()
                if len(friend_posts) > 10:
                    friend_posts = random.sample(friend_posts, 10)
            except:
                friend_posts = []
            friend_post_i.append(friend_posts)
        friends_post_list.append(friend_post_i)
        user_id_list.append(friends_relation.iloc[i, 2])
    selected_dict = {"user_id": user_id_list,
                     "neighbor_posts": friends_post_list,
                     }
    with open(output_file, 'wb') as f:
        pickle.dump(selected_dict, f)
        f.close()
    return selected_dict
