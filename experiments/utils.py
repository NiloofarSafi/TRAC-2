# -----------------------------------------------------------------------------------
# Earlier it was named as CommonUtils
# -----------------------------------------------------------------------------------
import sys
sys.path.append('../')

import json
import shutil

import joblib
import matplotlib
import numpy as np
import pandas as pd

import config as C

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('ticks')


def get_df_predictions_with_ground_truths(imdb_id_list, sorted_tag_idx_list):
    """
    Given the params, Write a csv file.

    imdb_id | ground_truth_tags | predicted_ranking

    :param imdb_id_list:
    :param predicted_probabilities:
    :return:
    """

    df_to_write = pd.DataFrame(columns=['imdb_id', 'ground_truth', 'predicted_ranking'])
    ground_truths = json.load(open('../../data/MPST/tag_assignment_data/movie_to_label_name.json', 'r'))
    idx_to_tag = {int(k): v for k, v in json.load(open(C.idx_to_tag_path, 'r')).items()}

    for imdb_id, sorted_tag_idx in zip(imdb_id_list, sorted_tag_idx_list):
        df_to_write.loc[len(df_to_write)] = [imdb_id, ', '.join(ground_truths[imdb_id]),
                                             ', '.join([idx_to_tag[idx] for idx in sorted_tag_idx])]

    return df_to_write


# ===================================================================
# Utilities for training
# ===================================================================
def check_setting_change_after_epoch(run_name):
    """
    Opens a file to check for command to stop or change learning rate of the optimizer
    :return:
    """
    tokens = open('run_settings.txt', 'r').read()
    print(tokens)
    tokens = tokens.split('|')
    r = None
    print(tokens, len(tokens))
    if len(tokens) < 2 or len(tokens) > 3:
        r = None

    elif run_name == tokens[0]:
        if len(tokens) == 3: # run_name|lr|0.01
            r = float(tokens[2])
        else:
            r = 'stop'

    if len(tokens) > 1 and run_name == tokens[0]:
        with open('run_settings.txt', 'w') as wf:
            wf.write('')
            wf.close()

    return r


def copy_files(file_list, dirpath):
    for f in file_list:
        shutil.copy(f, dirpath)


# Plotting during training
def get_heatmap_data(prediction_csv_file_path, limit_movies):
    df = pd.read_csv(prediction_csv_file_path, sep=',')
    heatmap_data = np.full((len(df), 71), 150)

    for idx, row in df.iterrows():
        gt = row['ground_truth'].split(', ')
        predicted_ranking = row['predicted_ranking'].split(', ')

        for t in gt:
            heatmap_data[idx, predicted_ranking.index(t)] = 0

    heatmap_data = np.transpose(heatmap_data[:limit_movies])

    return heatmap_data


def create_heatmap_of_ranking(prediction_csv_file_path, limit_movies):
    # Plot
    plt.figure(figsize=(20, 5))
    sns.heatmap(get_heatmap_data(prediction_csv_file_path, limit_movies), cmap='Dark2', square='True', cbar=False)
    plt.yticks([])
    plt.xticks([])

    plt.show()


def create_subplots_of_ranking(dump_dir, filename, limit_movies):
    fig, ax = plt.subplots(2, 1)
    fig.set_figwidth(15)
    fig.set_figheight(6)
    sns.heatmap(get_heatmap_data(dump_dir + 'train_predictions_last.csv', limit_movies), cmap='gray', square='True',
                        cbar=False, ax=ax[0])
    sns.heatmap(get_heatmap_data(dump_dir + 'validation_predictions_last.csv', limit_movies), cmap='gray', square='True',
                        cbar=False, ax=ax[1])

    ax[0].set_title('Train')
    ax[1].set_title('Validation')
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    plt.savefig(dump_dir + filename)


def format_individual_weights(doc_weight, sent_weight, sequence, idx2word):
    """
    Formats the attention weights by aligning with the sentences and words

    Parameters:
    -----------
    doc_weight:
        Vector containing weights of each sentence in the text
    sent_weight:
        2D vector containing weights of each word in each sentence
    sequence:
        2D padded sequence matrix of the document
    idx2word:
        Vocabulary dictionary

    Returns:
    --------
    sentence_weight_list:
        List containing floats representing weights for sentences
    formatted_words_weight_list:
        List of list where each index is a mini list where first index has a word and second index has the score

    """

    sentence_weight_list = []
    formatted_words_weight_list = []

    for i in range(len(doc_weight)):  # Iterate over sentences

        # Skip paddings
        if doc_weight[i] == 0 or (len(doc_weight) - len(sequence)) > i:
            continue

        # Create a list of [word: score]
        word_scores = [[idx2word[idx], float(score)] for idx, score in zip(sequence[i], sent_weight[i]) if idx > 0]

        # print(word_scores)
        sentence_weight_list.append(float(doc_weight[i]))
        formatted_words_weight_list.append(word_scores)

    return sentence_weight_list, formatted_words_weight_list


def format_and_dump_attention_weights(id_list, attention_weights_dict, id_title_tag_df, prediction_gt_df, rank_hit_scores):
    """
    Creates a dictionary containing the movie id, title, ground truth tags, predicted ranking of tags, attention weights
    for words and sentences in plot synopses, review titles and bodies.

    Paramters
    ----------
    id_list: List
        List of IMDB ids

    attention_weights_dict: Dict
        Dictionary containing weights list for different types of data.
        Keys
        -----
        * plot_doc_attention_weights
        * plot_sent_attention_weights
        * rb_doc_attention_weights
        * rb_sent_attention_weights
        * rt_doc_attention_weights
        * rt_sent_attention_weights
        * gate_weights

    id_title_tag_df: pandas.DataFrame
        Columns = [imdb id, title, ground truth tags]

    prediction_gt_df: pandas.DataFrame
        Columns = [imdb id, ground truth tags, predicted ranking of tags

    Returns
    -------
    A dictionary suitable for dumping as JSON file and show attention in HTML viewer

    """
    joblib.dump([id_list, attention_weights_dict, id_title_tag_df, prediction_gt_df],
                '../test_data/attn_weights_test.pkl')

    # Load index to word file
    idx2word = json.load(open(C.idx2w_file_path, 'r'))
    idx2word = {int(k): v for k, v in idx2word.items()}

    weight_dict = {}

    for i in range(len(id_list)):
        # Get necessary data from parameters
        imdb_id = id_list[i]
        movie_info_row = id_title_tag_df[id_title_tag_df['imdb_id'] == imdb_id]
        movie_title = movie_info_row['title'].to_string(index=False)
        gt_tags = movie_info_row['tags'].values.tolist()
        predicted_ranked_list = prediction_gt_df[prediction_gt_df['imdb_id'] == imdb_id]['predicted_ranking'].values.tolist()
        rank_score = rank_hit_scores[i]

        plot_doc_attention_weights = attention_weights_dict['plot_doc_attention_weights'][i]
        plot_sent_attention_weights = attention_weights_dict['plot_sent_attention_weights'][i]
        rb_doc_attention_weights = attention_weights_dict['rb_doc_attention_weights'][i]
        rb_sent_attention_weights = attention_weights_dict['rb_sent_attention_weights'][i]
        rt_doc_attention_weights = attention_weights_dict['rt_doc_attention_weights'][i]
        rt_sent_attention_weights = attention_weights_dict['rt_sent_attention_weights'][i]
        gate_weights = attention_weights_dict['gate_weights'][i].tolist()

        # Load the Feature JSON file
        feature_dict = json.load(open(C.feature_dump_path + imdb_id + '.json', 'r'))

        # Get the formatted weight lists for plot, review body and title
        plot_sentence_weight_list, plot_formatted_words_weight_list = format_individual_weights(
                                                                    plot_doc_attention_weights,
                                                                    plot_sent_attention_weights,
                                                                    feature_dict['plot_sequence_hierarchical_padded'],
                                                                    idx2word)

        rb_sentence_weight_list, rb_formatted_words_weight_list = format_individual_weights(
                                                            rb_doc_attention_weights,
                                                            rb_sent_attention_weights,
                                                            feature_dict['review_body_sequence_hierarchical_padded'],
                                                            idx2word)

        rt_sentence_weight_list, rt_formatted_words_weight_list = format_individual_weights(
                                                            rt_doc_attention_weights,
                                                            rt_sent_attention_weights,
                                                            feature_dict['review_title_sequence_hierarchical_padded'],
                                                            idx2word)

        weight_dict[imdb_id] = {'title'              : movie_title,
                                'tags'               : gt_tags,
                                'prediction'         : predicted_ranked_list,
                                'rank_hit_score'     : rank_score,

                                'doc_len'            : len(plot_sentence_weight_list),
                                'sentence_weights'   : plot_sentence_weight_list,
                                'words_weights'      : plot_formatted_words_weight_list,

                                'rb_doc_len'         : len(rb_sentence_weight_list),
                                'rb_sentence_weights': rb_sentence_weight_list,
                                'rb_words_weights'   : rb_formatted_words_weight_list,

                                'rt_doc_len'         : len(rt_sentence_weight_list),
                                'rt_sentence_weights': rt_sentence_weight_list,
                                'rt_words_weights'   : rt_formatted_words_weight_list,
                                'gate_weights'       : gate_weights
                                }

    with open('../test_data/attn_weight.json', 'w') as f:
        json.dump(weight_dict, f)

    return weight_dict


if __name__ == '__main__':

    id_list, attention_weights_dict, id_title_tag_df, prediction_gt_df = joblib.load('../test_data/attn_weights_test.pkl')
    format_and_dump_attention_weights(id_list, attention_weights_dict, id_title_tag_df, prediction_gt_df)

    #print(joblib.load('attn_weights_test.pkl').keys())
