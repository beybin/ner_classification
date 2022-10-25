import re

import pandas as pd

from config import max_seq_len, split_str, train_data_path


def get_split_list(text, split_str=split_str, max_seq_len=max_seq_len):
    # return re.split(r'([' + split_str + '])', text)
    split_list = (re.split(r'([' + split_str + '])', text))
    text_list = []
    split_text = ''
    for i in range(len(split_list)):
        tl = split_list[i]

        if len(split_text) + len(tl) < max_seq_len:
            split_text += tl
        else:
            text_list.append(split_text)
            split_text = tl

        if i == len(split_list) - 1:
            text_list.append(split_text)

    return text_list


def get_sent_cut(sents, tags):
    new_sents, new_tags = [], []
    tmp_sents, tmp_tags = [], []
    for i in range(len(sents)):
        if sents[i] in split_str:
            tmp_sents.append(sents[i])
            tmp_tags.append(tags[i])
            new_sents.append(tmp_sents)
            new_tags.append(tmp_tags)
            tmp_sents, tmp_tags = [], []
        else:
            tmp_sents.append(sents[i])
            tmp_tags.append(tags[i])
    return new_sents, new_tags


def get_d(sentence, tags):
    d, last_flag = [], ''
    for i in range(len(sentence)):
        char, this_flag = sentence[i], tags[i]
        if this_flag == 'O' and last_flag == 'O':
            d[-1][0] += char
        elif this_flag == 'O' and last_flag != 'O':
            d.append([char, 'O'])
        elif this_flag[:1] == 'B':
            d.append([char, this_flag[2:]])
        else:
            try:
                d[-1][0] += char
            except:
                print('tag error:', char, this_flag)
        last_flag = this_flag
    return d


def load_data():
    train_data = pd.read_csv(train_data_path)
    train_data['BIO_anno'] = train_data['BIO_anno'].apply(lambda x: x.split(' '))
    train_data['training_data'] = train_data.apply(lambda row: (list(row['text']), row['BIO_anno']), axis=1)

    D = []
    ind = -1
    class_list = train_data['class']
    for sentence, tags in train_data['training_data']:
        ind += 1
        text = ''.join(sentence)
        class_label = int(class_list[ind])

        D.append((get_d(sentence, tags), class_label))

        # 切分后，情感分类标签是否和原有一样？
        # if len(text) > max_seq_len - 2 and class_label != 2:
        #     continue
        # text = ''.join(sentence)
        # if len(text) > max_seq_len - 2:
        #     new_sentence, new_tags = get_sent_cut(sentence, tags)
        #     for i in range(len(new_sentence)):
        #         D.append((get_d(new_sentence[i], new_tags[i]), class_label))
        # else:
        #     D.append((get_d(sentence, tags), class_label))
    print('data len:', len(D))
    return D
