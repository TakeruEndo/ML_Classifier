"""
reference
https://medium.com/@aniruddha.choudhury94/part-2-bert-fine-tuning-tutorial-with-pytorch-for-text-classification-on-the-corpus-of-linguistic-18057ce330e1
https://qiita.com/yuki_uchida/items/09fda4c5c608a9f53d2f
"""

import os
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import time

from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from utils import tokenize_sentences, get_attention_mask, seed_everything, format_time, flat_f1_score, flat_accuracy, read_csv


BATCH_SIZE = 32
MAX_LEN = 128  # 64


def train(model, train_dataloader, validation_dataloader, epochs, scheduler, optimizer, criterion, kfold):
    # If there's a GPU available...
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    model.to(device)

    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    stock_f1_score = 0
    # For each epoch...
    for epoch_i in range(0, epochs):

        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(
                    step, len(train_dataloader), elapsed))

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask,
                            labels=b_labels)

            loss = outputs[0]
            total_loss += loss.item()
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epcoh took: {:}".format(
            format_time(time.time() - t0)))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.
        print("")
        print("Running Validation...")
        t0 = time.time()
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()
        # Tracking variables
        eval_accuracy, eval_f1_score = 0, 0
        nb_eval_steps = 0
        # Evaluate data for one epoch
        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)
            logits = outputs[0]
            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_f1_score = flat_f1_score(logits, label_ids)
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy
            eval_f1_score += tmp_eval_f1_score
            # Track the number of batches
            nb_eval_steps += 1
        if eval_f1_score > stock_f1_score:
            torch.save(model.state_dict(), './bert_models/exp5/epoch_' + str(epoch_i) +
                       '_fold_' + str(kfold) + '_train.pth')
        # Report the final accuracy for this validation run.
        print("  accuracy: {0:.2f}".format(eval_accuracy / nb_eval_steps))
        print("  F1_score: {0:.2f}".format(eval_f1_score / nb_eval_steps))
        print("  Validation took: {:}".format(format_time(time.time() - t0)))
    print("")
    print("Training complete!")

    return model, loss_values


def main(i):
    seed_everything(2015)
    df = read_csv('train_5fold.csv')
    df[TARGET] = df[TARGET].astype(int) - 1

    # 学習用と評価用にデータを分割
    train_df = df.query('kfold != {}'.format(i))
    test_df = df.query('kfold == {}'.format(i))

    train_sentences, train_inputs, train_labels = tokenize_sentences(
        train_df, 'train', TARGET, SENTENCE, MAX_LEN)
    train_masks = get_attention_mask(train_inputs)
    valid_sentences, validation_inputs, validation_labels = tokenize_sentences(
        test_df, 'train', TARGET, SENTENCE, MAX_LEN)
    validation_masks = get_attention_mask(validation_inputs)

    # Convert all inputs and labels into torch tensors, the required datatype
    # for our model.
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(
        validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(
        validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    model = BertForSequenceClassification.from_pretrained(
        # Use the 12-layer BERT model, with an uncased vocab.
        "bert-base-uncased",
        # The number of output labels--2 for binary classification.
        num_labels=4,
        # You can increase this for multi-class tasks.
        # Whether the model returns attentions weights.
        output_attentions=False,
        # Whether the model returns all hidden-states.
        output_hidden_states=False
    )

    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )
    criterion = torch.nn.CrossEntropyLoss()
    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 7
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)

    train(model=model, train_dataloader=train_dataloader, validation_dataloader=validation_dataloader,
          epochs=epochs, scheduler=scheduler, optimizer=optimizer, criterion=criterion, kfold=i)
