import pandas as pd
import numpy as np

import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from utils import tokenize_sentences, get_attention_mask, seed_everything, format_time, flat_f1_score, flat_accuracy, read_csv


BATCH_SIZE = 32
MAX_LEN = 128  # 64


def eval(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    predictions = []
    for batch in dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)
        logits = outputs[0]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        # label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        # true_labels.append(label_ids)

    return predictions


def main():
    df = read_csv('test.csv')
    weight_path = 'src/bert/models/epoch_7_train.pth'

    sentences, input_ids = tokenize_sentences(
        df, 'test', TARGET, SENTENCE, MAX_LEN)
    attention_masks = get_attention_mask(input_ids)

    # Convert to tensors.
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)

    batch_size = 32
    # Create the DataLoader.
    prediction_data = TensorDataset(prediction_inputs, prediction_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(
        prediction_data, sampler=prediction_sampler, batch_size=batch_size)

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

    model.load_state_dict(torch.load(weight_path))

    predictions = eval(model, prediction_dataloader)

    # Combine the predictions for each batch into a single list of 0s and 1s.
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

    submit = pd.DataFrame({'index': df['id'], 'pred': flat_predictions + 1})
    submit.to_csv("submit_model_bert.csv", index=False, header=False)


if __name__ == '__main__':
    main()
