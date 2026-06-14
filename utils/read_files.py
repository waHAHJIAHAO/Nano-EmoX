import os
import json
import math
import random
import numpy as np
import pandas as pd


def func_shuffle_list_data(whole_json):
    indices = np.arange(len(whole_json))
    random.shuffle(indices)
    new_json = []
    for index in indices:
        new_json.append(whole_json[index])
    return new_json


def func_split_list_data(data, store_root, split_num=8, shuffle=True):
    if not os.path.exists(store_root):
        os.makedirs(store_root)
    if shuffle:
        data = func_shuffle_list_data(data)
    subset_number = math.ceil(len(data) / split_num)
    for ii in range(split_num):
        sub_data = data[ii * subset_number:(ii + 1) * subset_number]
        save_path = os.path.join(store_root, f'split-{ii}.npy')
        np.save(save_path, sub_data)


def func_read_key_from_csv(csv_path, key):
    values = []
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        if key not in row:
            values.append("")
        else:
            value = row[key]
            if pd.isna(value):
                value = ""
            values.append(value)
    return values


def func_write_key_to_csv(csv_path, names, name2key, keynames):
    if len(name2key) == 0 or len(keynames) == 0:
        df = pd.DataFrame(data=names, columns=['name'])
        df.to_csv(csv_path, index=False)
        return
    if isinstance(keynames, str):
        keynames = [keynames]
    assert isinstance(keynames, list)
    columns = ['name'] + keynames
    values = []
    for name in names:
        value = name2key[name]
        values.append(value)
    values = np.array(values)
    if len(values.shape) == 1:
        assert len(keynames) == 1
    else:
        assert values.shape[-1] == len(keynames)
    data = np.column_stack([names, values])
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(csv_path, index=False)


def func_read_text_file(file_path):
    try:
        with open(file_path, encoding='utf8') as f:
            lines = [line.strip() for line in f]
        lines = [line for line in lines if len(line) != 0]
        return lines
    except:
        with open(file_path, encoding='ansi') as f:
            lines = [line.strip() for line in f]
        lines = [line for line in lines if len(line) != 0]
        return lines
