import os
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import time
import datetime

from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score
import torch
from transformers import BertTokenizer


def seed_everything(seed):
    """for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Function to calculate the accuracy of our predictions vs labels
def flat_f1_score(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, pred_flat, average='macro')
    # return np.sum(pred_flat == labels_flat) / len(labels_flat)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def read_csv(path):
    return pd.read_csv(path)


def tokenize_sentences(df, type, target, sentence, max_len):
    """sentenecs to IDs(encode)
    1. 文章をトークンに分割する
    2. 先頭と末尾に[CLS] [SEP]トークンを挿入する
    3. トークンをIDsに変換する

    Args:
        df (DataFrame): embeddingする対象

    Returns:
        [list]: 変換元の文章とエンコード後の文章 
    """
    print('Loading BERT tokenizer...')
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased', do_lower_case=True)
    # Get the lists of sentences and their labels.
    sentences = df[sentence].values

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    # For every sentence...
    for sent in sentences:
        encoded_sent = tokenizer.encode(
            sent,                      # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            # This function also supports truncation and conversion
            # to pytorch tensors, but we need to do padding, so we
            # can't use these features :( .
            # max_length = 128,          # Truncate all sentences.
            # return_tensors = 'pt',     # Return pytorch tensors.
        )
        input_ids.append(encoded_sent)
    print('Original: ', sentences[0])
    print('Token IDs:', input_ids[0])

    #  Padding & Truncating
    print('Max sentence length: ', max([len(sen) for sen in input_ids]))
    # Set the maximum sequence length.
    # I've chosen 64 somewhat arbitrarily. It's slightly larger than the
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long",
                              truncating="post", padding="post")

    if type == 'train':
        labels = df[target].values
        return sentences, input_ids, labels
    return sentences, input_ids


def get_attention_mask(input_ids):
    """Attention_mask
    tokenのなかで、どれが実際のごくで、どれがパディングかを明示する
    """
    attention_masks = []
    # For each sentence...
    for sent in input_ids:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
    return attention_masks
