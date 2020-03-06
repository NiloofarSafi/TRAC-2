from __future__ import division
import nltk
import re
from collections import Counter
import json
import numpy as np
import pandas as pd
import os
import statistics
from transformers import *
import sys
sys.path.append('../')
from torchmoji.sentence_tokenizer import SentenceTokenizer
from experiments import config as C
from torchmoji.word_generator import WordGenerator

wordgen = WordGenerator(None, allow_unicode_text=True,
                             ignore_emojis=False,
                             remove_variation_selectors=True,
                             break_replacement=True)

print('Tokenizing using dictionary from {}'.format(C.VOCAB_PATH))
with open(C.VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)

tokenizer = SentenceTokenizer(vocabulary, C.maxlen)

# tokenizer = nltk.tokenize.TweetTokenizer()

def tokenization(post):
    wordgen.reset_stats()
    wordgen.stream = [post]
    for s_words, s_info in wordgen:
        tokens = s_words
    return tokens


def one_hot(idx, length):
    """
    create one-hot vectors
    :param idx: non-zero element
    :param length: size of label space
    :return:
    """
    output = [0] * length
    output[idx] = 1
    return output


# ------------------------------------------------------------------------
# Preprocessing
#-------------------------------------------------------------------------
def replacements(text):
    text = re.sub(r'[@＠][a-zA-Z0-9_.!?-\\]+', "@username", text.rstrip())
    # text = re.sub(r"((www\.[^\s]+)|(https?:\/\/[^\s]+))", "url", text.rstrip())
    text = re.sub(r'https?://|www\.', "url", text.rstrip())
    text = text.replace('-', ' ')
    # text = text.replace('_', ' ')
    # text = text.replace('#', ' ')
    return text


def posts_to_BERT_feature(path, data_partition):

    data_folder = '../preprocessed_data'

    b_tokenizer = BertTokenizer.from_pretrained(C.bert_model, do_lower_case=True)

    data = pd.read_csv(path, delimiter=',', encoding='utf-8', lineterminator='\n')

    feature_list = {}

    for ii, row in data.iterrows():
        # print("{}: {}".format(row['ID'], row['Text']))
        if type(row['Text']) != float:
            text = row['Text'].strip('\r\n')

            text = replacements(text)
            if text == ' ':
                text = '-'
            q_tokenized = b_tokenizer.encode(text, add_special_tokens=True, max_length=300)
            try:
                text_dm, _, _ = tokenizer.tokenize_sentences([text])

            except:
                print("{}: {}\n{}".format(row['ID'], row['Text'], text))
                exit()
            # tokenized = tokenizer.tokenize(question.lower())
            # q_masked = ['DD' if token.isdigit() else token for token in tokenized]


            features = {
                'id': row['ID'],
                'post':text,
                'tokenized': q_tokenized,
                'DM': text_dm.tolist()[0],
                'label': row['Sub-task A'],
                'gen': row['Sub-task B'],
                'y1': one_hot(C.label2idx[row['Sub-task A']], 3),
                'y2': one_hot(C.gen2idx[row['Sub-task B']], 2)
            }


            feature_list[row['ID']] = features

    file_name = os.path.join(data_folder, '{}.json'.format(data_partition))
    json.dump(feature_list, open(file_name, mode='w'))
    print('Preprocessed preprocessed_data dumped for {} set!'.format(data_partition))


if __name__ == '__main__':

    posts_to_BERT_feature('../eng/trac2_eng_train_final.csv', 'eng_train')

