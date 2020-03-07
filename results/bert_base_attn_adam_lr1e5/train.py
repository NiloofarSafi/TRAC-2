__author__ = "Niloofer Safi Samghabadi"
import os
import sys

sys.path.append('../')
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch
import torch.utils.data

torch.manual_seed(3)
if torch.cuda.\
        is_available():
    torch.cuda.manual_seed_all(7)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from torch import nn
import time
import json
import numpy as np
from torch import optim
from torch.nn import functional as F
from experiments.TorchHelper import TorchHelper
from experiments.tf_logger import Logger
import utils as U
from models.model import *
import config as C
import shutil
from sklearn.metrics import f1_score
import warnings
import copy
from nltk.tokenize import TweetTokenizer
import random
from torchmoji.word_generator import WordGenerator


warnings.filterwarnings('ignore')


random.seed(3)
np.random.seed(3)

torch_helper = TorchHelper()

tokenizer = TweetTokenizer()
wordgen = WordGenerator(None, allow_unicode_text=True,
                             ignore_emojis=False,
                             remove_variation_selectors=True,
                             break_replacement=True)

#NAG, OAG, CAG
task1_weights = [0.53151412, 1.78422392, 1.79174652]
#NGEN, GEN
task2_weights = [0.58438203, 3.46271605]

class_weights1 = torch.FloatTensor(task1_weights)
class_weights2 = torch.FloatTensor(task2_weights)

start_epoch = 0
batch_size = C.batch_size
max_epochs = 200
learning_rate = 0.00001
optimizer_type = 'adam'

tokenizer = TweetTokenizer()

l2_regularize = True
l2_lambda = 0.01

alphabet_path = "alphabet.json"

# Creates the directory where the results, logs, and models will be dumped.

run_name = 'bert_base_attn_adam_lr1e5'

description = ''

output_dir_path = '../results/' + run_name + '/'
if not os.path.exists(output_dir_path):
    os.mkdir(output_dir_path)
logger = Logger(output_dir_path + 'logs')

# Files to keep backup
backup_file_list = [os.path.dirname(os.path.realpath(__file__)) + '/train.py',
                    os.path.dirname(os.path.realpath(__file__)) + '/../models/model.py']


def copy_files(file_list, dirpath):
    for f in file_list:
        shutil.copy(f, dirpath)


run_mode = 'train'
# ----------------------------------------------------------------------------
# Load Data
# ----------------------------------------------------------------------------

features_train = json.load(open('../preprocessed_data/eng_train.json'))
features_dev = json.load(open('../preprocessed_data/eng_dev.json'))

train_set = [val for key,val in features_train.items()]
print('Train Loaded')

validation_set = [val for key,val in features_dev.items()]
print('Validation Loaded')

print('Data Split: Train (%d), Dev (%d)' % (len(train_set), len(validation_set)))

# ----------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------

def create_model():
    """
    Creates and returns the model.
    Moves to GPU if found any.
    :return:

    """

    model = BERTRA()

    model.cuda()
    if run_mode == 'resume':
        torch_helper.load_saved_model(model, output_dir_path + 'best.pth')
        print('model loaded')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    return model


def compute_l2_reg_val(model):
    if not l2_regularize:
        return 0.

    l2_reg = None

    for w in model.parameters():
        if l2_reg is None:
            l2_reg = w.norm(2)
        else:
            l2_reg = l2_reg + w.norm(2)

    return l2_lambda * l2_reg.item()

# ----------------------------------------------------------------------------
# Padding
# ----------------------------------------------------------------------------

def pad_features(docs_ints, seq_length=200):

    # getting the correct rows x cols shape
    features = np.zeros((len(docs_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(docs_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features

def masking(docs_ints, seq_length=200):

    # getting the correct rows x cols shape
    masks = np.zeros((len(docs_ints), seq_length), dtype=int)

    # for each review, I grab that review and
    for i, row in enumerate(docs_ints):
        #mask[i, :len(row)] = 1
        masks[i, -len(row):] = 1

    return masks

# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------

def train(model, optimizer, shuffled_train_set):
    """
    Trains the model using the optimizer for a single epoch.
    :param model: pytorch model
    :param optimizer:
    :return:
    """

    start_time = time.time()

    model.train()

    batch_idx = 1
    total_loss = 0
    batch_x, batch_t, batch_y1, batch_y2 = [], [], [], []
    random.Random(1234).shuffle(shuffled_train_set)

    for i in range(len(shuffled_train_set)):

        batch_x.append(shuffled_train_set[i]['tokenized'])
        batch_t.append(shuffled_train_set[i]['DM'])

        batch_y1.append(shuffled_train_set[i]['y1'])
        batch_y2.append(shuffled_train_set[i]['y2'])

        if len(batch_x) == batch_size or i == len(shuffled_train_set) - 1:

            optimizer.zero_grad()

            mask = masking(batch_x)
            padded = pad_features(batch_x)

            batch_emoj = np.array(batch_t, dtype='uint16')
            batch_emoj = torch.from_numpy(batch_emoj.astype('int64')).long()

            out = model(torch.tensor(padded).cuda(), torch.tensor(mask).cuda())

            y_pred1 = out['y_pred1'].cpu()
            y_pred2 = out['y_pred2'].cpu()
            loss = F.binary_cross_entropy(y_pred1, torch.Tensor(batch_y1), weight=class_weights1) + \
                    F.binary_cross_entropy(y_pred2, torch.Tensor(batch_y2), weight=class_weights2) +\
                    compute_l2_reg_val(model)

            total_loss += loss.item()
            loss.backward()

            optimizer.step()

            torch_helper.show_progress(batch_idx , np.ceil(len(shuffled_train_set) / batch_size), start_time,
                                   round(total_loss / (i+1), 4))
            batch_idx += 1
            batch_x, batch_t, batch_y1, batch_y2 = [], [], [], []

    return model, shuffled_train_set

# ----------------------------------------------------------------------------
# Evaluate the model
# ----------------------------------------------------------------------------
def evaluate(model, dev_set):

    model.eval()

    total_loss = 0
    batch_x, batch_t, batch_y1, batch_y2 = [], [], [], []
    y1_true, y2_true = [], []

    label_predictions, gender_predictions = [], []

    with torch.no_grad():
        for i in range(len(dev_set)):

            batch_x.append(dev_set[i]['tokenized'])
            batch_t.append(dev_set[i]['DM'])

            batch_y1.append(dev_set[i]['y1'])
            batch_y2.append(dev_set[i]['y2'])


            y1_true.append(dev_set[i]['y1'])
            y2_true.append(dev_set[i]['y2'])

            if len(batch_x) == batch_size or i == len(dev_set) - 1:
                mask = masking(batch_x)
                padded = pad_features(batch_x)

                batch_emoj = np.array(batch_t, dtype='uint16')
                batch_emoj = torch.from_numpy(batch_emoj.astype('int64')).long()


                # out = model(torch.tensor(padded).cuda(), batch_emoj.cuda(), torch.tensor(mask).cuda())
                out = model(torch.tensor(padded).cuda(), torch.tensor(mask).cuda())

                y_pred1 = out['y_pred1'].cpu()
                y_pred2 = out['y_pred2'].cpu()

                label_predictions.extend(list(torch.argmax(y_pred1, -1).numpy()))
                gender_predictions.extend(list(torch.argmax(y_pred2, -1).numpy()))

                loss = F.binary_cross_entropy(y_pred1, torch.Tensor(batch_y1), weight=class_weights1) + \
                       F.binary_cross_entropy(y_pred2, torch.Tensor(batch_y2), weight=class_weights2)


                total_loss += loss.item()

                batch_x, batch_t, batch_y1, batch_y2 = [], [], [], []

    macro_f1_1 = f1_score(y1_true, label_predictions, average='macro')
    macro_f1_2 = f1_score(y2_true, gender_predictions, average='macro')

    return label_predictions, \
           gender_predictions, \
           total_loss/len(dev_set), \
           macro_f1_1,\
           macro_f1_2



def training_loop():
    """

    :return:
    """
    model = create_model()


    if optimizer_type == 'adamW':
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    shuffled_train_set = train_set

    for epoch in range(start_epoch, max_epochs):

        # for p in model.tm.parameters():
        #     p.requires_grad = False

        for p in model.bert.parameters():
            p.requires_grad = False


        print('[Epoch %d] / %d : %s' % (epoch + 1, max_epochs, run_name))

        model, shuffled_train_set = train(model, optimizer, shuffled_train_set)

        print("Training Done!")

        val_label_pred, val_gender_pred, val_loss, val_label_f1, val_gender_f1 = evaluate(model, validation_set)
        train_label_pred, train_gender_pred, train_loss, train_label_f1, train_gender_f1 = evaluate(model, train_set)

        print('Evaluation Done!')

        current_lr = 0
        for pg in optimizer.param_groups:
            current_lr = pg['lr']

        print('Training Loss %.5f, Validation Loss %.5f' % (train_loss, val_loss))
        print('Training Label Macro F1 %.5f, Validation Label Macro F1 %.5f' % (train_label_f1, val_label_f1))
        print('Training Gender Macro F1 %.5f, Validation Gender Macro F1 %.5f' % (train_gender_f1, val_gender_f1))
        # print('Learning Rate', current_lr)


        is_best = torch_helper.checkpoint_model(model, optimizer, output_dir_path, val_label_f1, epoch + 1,
                                                'max')


        # -------------------------------------------------------------
        # Tensorboard Logging
        # -------------------------------------------------------------
        info = {'training loss'  : train_loss,
                'validation loss': val_loss,
                'train_label_f1': train_label_f1,
                'val_label_f1'  : val_label_f1,
                'train_gender_f1': train_gender_f1,
                'val_gender_f1': val_gender_f1,
                'lr'      : current_lr
                }

        for tag, value in info.items():
            logger.log_scalar(tag, value, epoch + 1)

        # Log values and gradients of the model parameters
        for tag, value in model.named_parameters():
            if value.grad is not None:
                tag = tag.replace('.', '/')

                if torch.cuda.is_available():
                    logger.log_histogram(tag, value.data.cpu().numpy(), epoch + 1)
                    logger.log_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)
                else:
                    logger.log_histogram(tag, value.data.numpy(), epoch + 1)
                    logger.log_histogram(tag + '/grad', value.grad.data.numpy(), epoch + 1)


if __name__ == '__main__':
    U.copy_files(backup_file_list, output_dir_path)
    with open(output_dir_path + 'description.txt', 'w') as f:
        f.write(description)
        f.close()

    training_loop()

