# author = "Niloofer"

from googletrans import Translator
import time
import nltk
import re
import pandas as pd
import json
import emoji
import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

tokenizer = nltk.tokenize.TweetTokenizer()
sentence_tokenizer = nltk.tokenize.sent_tokenize



def tokenize_msg(msg):
    """
    Tokenizes at word level.

    Parameters:
    -----------
    msg : String
        The string to tokenize

    Returns:
    --------
    A list of word tokens

    # >>> tokenize_msg('This is a message :)')
    ['This', 'is', 'a', 'message', ':)']

    """
    return tokenizer.tokenize(msg)



def translate_(file_path, dest_lang):
    """
    Translate the text in ``file_path'' file to the ``dest_lang'' language and dumped the results to ``dump_path''
    :param file_path:
    :param dest_lang:
    :return:
    """

    dump_path = file_path.split('.')[0] + '-' + dest_lang + '.json'

    docs = {}
    translator = Translator()
    df = pd.read_csv(file_path, sep=',', encoding='utf-8')

    counter = 1
    for idx, row in df.iterrows():

        text_no_emoj = give_emoji_free_text(row['Text'])

        try:
           translation = translator.translate(text=text_no_emoj, dest=dest_lang)
           print(idx)
           docs[row['ID']] = translation.text

        except:
           print('\n--------\n[NOT TRANSLATED] {}\n--------\n'.format(row['Text']))

        if counter%10 == 0:
            time.sleep(60)

        counter += 1

    json.dump(docs, open(dump_path, 'w'))

def add_translation(org, tran, dest_lang):
    """
    add translation for each row of preprocessed_data in original file based on the json file
    :param org: original preprocessed_data file
    :param tran: json file including translations
    :param dest_lang: destination language
    :return:
    """

    translations = json.load(open(tran, 'r'))
    data = pd.read_csv(org, encoding='utf-8')
    translated = []
    for idx, row in data.iterrows():
        if row['ID'] in translations:
            translated.append(translations[row['ID']])
        else:
            translated.append('EMPTY')

    new_column = dest_lang + "_translation"
    data[new_column] = translated

    dump_file = org.split('.')[0] + '_translated.csv'
    data.to_csv(dump_file)


def give_emoji_free_text(text):
    """
    remove the emojis from the text
    :param text:
    :return:
    """
    allchars = [str for str in text]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]

    for emoj in emoji_list:
        text = text.replace(emoj, '')

    clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    return clean_text


def augment_data(dest_lang):

    lang_dict = {'en':'eng', 'hi':'hin', 'bn':'iben'}
    languages = [val for key, val in lang_dict.items()]

    main_file = lang_dict[dest_lang]+'/trac2_'+lang_dict[dest_lang]+'_train.csv'
    main_data = pd.read_csv(main_file, encoding='utf-8')

    translation_col = dest_lang+'_translation'

    for lang in languages:
        if lang != lang_dict[dest_lang]:
            for option in ['_train_translated.csv', '_dev_translated.csv']:
                t_file = lang+'/trac2_'+lang+option
                translation = pd.read_csv(t_file, encoding='utf-8')
                for idx, row in translation.iterrows():
                    main_data = main_data.append({'ID': row['ID'], 'Text': row[translation_col], 'Sub-task A': row['Sub-task A'], 'Sub-task B': row['Sub-task B']}, ignore_index=True)

    output_file = lang_dict[dest_lang]+'/trac2_'+lang_dict[dest_lang]+'_train_final.csv'
    main_data.to_csv(output_file)



if __name__ == '__main__':

    # Languages >> English: "en", Hindi: "hi", Bengali: "bn"

    #translate_('iben/trac2_iben_train.csv', 'en')

    #add_translation('iben/trac2_iben_dev.csv', 'iben/trac2_iben_dev-en.json', 'en')

    augment_data('en')

