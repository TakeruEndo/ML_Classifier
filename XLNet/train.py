import os
import math

import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, XLNetTokenizer, XLNetModel, XLNetLMHeadModel, XLNetConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from model import XLNetForMultiLabelSequenceClassification


def tokenize_inputs(text_list, tokenizer, num_embeddings=512):
    """
    Tokenizes the input text input into ids. Appends the appropriate special
    characters to the end of the text to denote end of sentence. Truncate or pad
    the appropriate sequence length.
    """
    # tokenize the text, then truncate sequence to the desired length minus 2 for
    # the 2 special characters
    tokenized_texts = list(map(lambda t: tokenizer.tokenize(t)[
                           :num_embeddings-2], text_list))
    # convert tokenized text into numeric ids for the appropriate LM
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # append special token "<s>" and </s> to end of sentence
    input_ids = [tokenizer.build_inputs_with_special_tokens(
        x) for x in input_ids]
    # pad sequences
    input_ids = pad_sequences(
        input_ids, maxlen=num_embeddings, dtype="long", truncating="post", padding="post")
    return input_ids


def create_attn_masks(input_ids):
    """
    Create attention masks to tell model whether attention should be applied to
    the input id tokens. Do not want to perform attention on padding tokens.
    """
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    return attention_masks


def train(model, num_epochs,
          optimizer,
          train_dataloader, valid_dataloader,
          model_save_path,
          train_loss_set=[], valid_loss_set=[],
          lowest_eval_loss=None, start_epoch=0,
          device="cpu"
          ):
    """
    Train the model and save the model with the lowest validation loss
    """

    model.to(device)

    # trange is a tqdm wrapper around the normal python range
    for i in trange(num_epochs, desc="Epoch"):
        # if continue training from saved model
        actual_epoch = start_epoch + i

        # Training

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        tr_f1 = 0
        num_train_samples = 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            loss, f1 = model(
                b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            # store train loss
            tr_loss += loss.item()
            tr_f1 += f1
            num_train_samples += b_labels.size(0)
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # scheduler.step()

        # Update tracking variables
        epoch_train_loss = tr_loss/num_train_samples
        epoch_train_f1 = tr_f1/num_train_samples
        train_loss_set.append(epoch_train_loss)

        print("Train loss: {}".format(epoch_train_loss))
        print("Train f1: {}".format(epoch_train_f1))

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss = 0
        eval_f1 = 0
        num_eval_samples = 0

        # Evaluate data for one epoch
        for batch in valid_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate validation loss
                loss, f1 = model(
                    b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                # store valid loss
                eval_loss += loss.item()
                eval_f1 += f1

                num_eval_samples += b_labels.size(0)

        epoch_eval_loss = eval_loss / num_eval_samples
        epoch_eval_f1 = eval_f1 / num_eval_samples
        valid_loss_set.append(epoch_eval_loss)

        print("Valid loss: {}".format(epoch_eval_loss))
        print("Valid f1: {}".format(epoch_eval_f1))

        if lowest_eval_loss == None:
            lowest_eval_loss = epoch_eval_loss
            highest_eval_loss = 0
            # save model
            save_model(model, model_save_path, actual_epoch,
                       lowest_eval_loss, train_loss_set, valid_loss_set)
        else:
            if epoch_eval_loss < lowest_eval_loss:
                lowest_eval_loss = epoch_eval_loss
                # save model
                save_model(model, model_save_path, actual_epoch,
                           lowest_eval_loss, train_loss_set, valid_loss_set)
            if epoch_eval_f1 > highest_eval_loss:
                highest_eval_loss = epoch_eval_f1
                save_model(model, os.path.join('best_f1_xlnet_models.bin'), actual_epoch,
                           lowest_eval_loss, train_loss_set, valid_loss_set)
        print("\n")

    return model, train_loss_set, valid_loss_set


def save_model(model, save_path, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist):
    """
    Save the model to the path directory provided
    """
    model_to_save = model.module if hasattr(model, 'module') else model
    checkpoint = {'epochs': epochs,
                  'lowest_eval_loss': lowest_eval_loss,
                  'state_dict': model_to_save.state_dict(),
                  'train_loss_hist': train_loss_hist,
                  'valid_loss_hist': valid_loss_hist
                  }
    torch.save(checkpoint, save_path)
    print("Saving model at epoch {} with validation loss of {}".format(epochs,
                                                                       lowest_eval_loss))
    return


def load_model(save_path):
    """
    Load the model from the path directory provided
    """
    checkpoint = torch.load(save_path)
    model_state_dict = checkpoint['state_dict']
    model = XLNetForMultiLabelSequenceClassification(
        num_labels=model_state_dict["classifier.weight"].size()[0])
    model.load_state_dict(model_state_dict)

    epochs = checkpoint["epochs"]
    lowest_eval_loss = checkpoint["lowest_eval_loss"]
    train_loss_hist = checkpoint["train_loss_hist"]
    valid_loss_hist = checkpoint["valid_loss_hist"]

    return model, epochs, lowest_eval_loss, train_loss_hist, valid_loss_hist


def generate_predictions(model, df, num_labels, device="cpu", batch_size=32):
    num_iter = math.ceil(df.shape[0]/batch_size)

    pred_probs = np.array([]).reshape(0, num_labels)

    model.to(device)
    model.eval()

    for i in range(num_iter):
        df_subset = df.iloc[i*batch_size:(i+1)*batch_size, :]
        X = df_subset["features"].values.tolist()
        masks = df_subset["masks"].values.tolist()
        X = torch.tensor(X)
        masks = torch.tensor(masks, dtype=torch.long)
        X = X.to(device)
        masks = masks.to(device)
        with torch.no_grad():
            logits = model(input_ids=X, attention_mask=masks)
            logits = logits.sigmoid().detach().cpu().numpy()
            pred_probs = np.vstack([pred_probs, logits])

    return pred_probs


def main():
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')

    tokenizer = XLNetTokenizer.from_pretrained(
        'xlnet-base-cased', do_lower_case=True)

    train_text_list = train[SENTENCE].values
    test_text_list = test[SENTENCE].values

    train_input_ids = tokenize_inputs(
        train_text_list, tokenizer, num_embeddings=250)
    test_input_ids = tokenize_inputs(
        test_text_list, tokenizer, num_embeddings=250)

    train_attention_masks = create_attn_masks(train_input_ids)
    test_attention_masks = create_attn_masks(test_input_ids)

    # add input ids and attention masks to the dataframe
    train["features"] = train_input_ids.tolist()
    train["masks"] = train_attention_masks

    test["features"] = test_input_ids.tolist()
    test["masks"] = test_attention_masks

    # train valid split
    train, valid = train_test_split(train, test_size=0.2, random_state=42)

    # Convert all of our input ids and attention masks into
    # torch tensors, the required datatype for our model
    X_train = train['features']
    Y_train = train[TARGET].astype(int) - 1
    X_valid = valid['features']
    Y_valid = valid[TARGET].astype(int) - 1

    train_masks = train['masks']
    valid_masks = valid['masks']

    X_train = torch.tensor(list(X_train.values))
    X_valid = torch.tensor(list(X_valid.values))

    # Y_train = torch.tensor(list(Y_train.values), dtype=torch.float32)
    Y_train = torch.eye(4)[list(Y_train.values)]
    # Y_valid = torch.tensor(list(Y_valid.values), dtype=torch.float32)
    Y_valid = torch.eye(4)[list(Y_valid.values)]

    train_masks = torch.tensor(list(train_masks.values), dtype=torch.long)
    valid_masks = torch.tensor(list(valid_masks.values), dtype=torch.long)

    # Select a batch size for training
    batch_size = 16

    # Create an iterator of our data with torch DataLoader. This helps save on
    # memory during training because, unlike a for loop,
    # with an iterator the entire dataset does not need to be loaded into memory

    train_data = TensorDataset(X_train, train_masks, Y_train)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=batch_size)

    validation_data = TensorDataset(X_valid, valid_masks, Y_valid)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data,
                                       sampler=validation_sampler,
                                       batch_size=batch_size)

    model = XLNetForMultiLabelSequenceClassification(num_labels=4)

    optimizer = AdamW(model.parameters(), lr=2e-5,
                      weight_decay=0.01, correct_bias=False)

    num_epochs = 10

    cwd = os.getcwd()
    model_save_path = os.path.join(cwd, 'xlnet_models.bin')
    model, train_loss_set, valid_loss_set = train(model=model,
                                                  num_epochs=num_epochs,
                                                  optimizer=optimizer,
                                                  train_dataloader=train_dataloader,
                                                  valid_dataloader=validation_dataloader,
                                                  model_save_path=model_save_path,
                                                  device='cuda')

    num_labels = 4
    weight_path = 'best_f1_xlnet_models.bin'
    model.load_state_dict(torch.load(weight_path))
    pred_probs = generate_predictions(
        model, test, num_labels, device="cuda", batch_size=32)
    flat_predictions = np.argmax(pred_probs, axis=1).flatten()
    submit = pd.DataFrame({'index': test['id'], 'pred': flat_predictions + 1})
    submit.to_csv("submit_model_xlnet_4_best_f1.csv",
                  index=False, header=False)


if __name__ == '__main__':
    main()
