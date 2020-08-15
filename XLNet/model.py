import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, XLNetTokenizer, XLNetModel, XLNetLMHeadModel, XLNetConfig
import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import f1_score


class XLNetForMultiLabelSequenceClassification(torch.nn.Module):

    def __init__(self, num_labels=4):
        super(XLNetForMultiLabelSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased')
        self.classifier = torch.nn.Linear(768, num_labels)

        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, labels=None):
        # last hidden layer
        last_hidden_state = self.xlnet(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)
        # pool the outputs into a mean vector
        mean_last_hidden_state = self.pool_hidden_state(last_hidden_state)
        logits = self.classifier(mean_last_hidden_state)

        if labels is not None:
            # labels = torch.eye(10)[label]
            # labels = labels.view(-1, self.num_labels))
            loss_fct = BCEWithLogitsLoss()
            logits = logits.view(-1, self.num_labels)
            loss = loss_fct(logits, labels)
            f1 = self.flat_f1_score(logits, labels)
            return loss, f1
        else:
            return logits

    def flat_f1_score(self, preds, labels):
        preds = preds.detach().cpu().numpy()
        labels = labels.to('cpu').numpy()
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = np.argmax(labels, axis=1).flatten()
        return f1_score(labels_flat, pred_flat, average='macro')

    def freeze_xlnet_decoder(self):
        """
        Freeze XLNet weight parameters. They will not be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = False

    def unfreeze_xlnet_decoder(self):
        """
        Unfreeze XLNet weight parameters. They will be updated during training.
        """
        for param in self.xlnet.parameters():
            param.requires_grad = True

    def pool_hidden_state(self, last_hidden_state):
        """
        Pool the output vectors into a single mean vector 
        """
        last_hidden_state = last_hidden_state[0]
        mean_last_hidden_state = torch.mean(last_hidden_state, 1)
        return mean_last_hidden_state
