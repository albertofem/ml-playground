import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils.class_weight import compute_class_weight
import argparse
import os

parser = argparse.ArgumentParser(description='Argument Parser')

parser.add_argument('--training_data_file', type=str, required=True, help='Location of the training data')
parser.add_argument('--model_output_dir', type=str, required=True, help='Directory to output the model')
parser.add_argument('--transformer_cache_dir', type=str, required=True, help='Directory to download transformers model data cache')
parser.add_argument('--bert_revision', type=str, required=True, help='Revision for the BERT model (commit or branch)')

args = parser.parse_args()


class BertSpamDetectorModel(nn.Module):
    def __init__(self, bert):
        super(BertSpamDetectorModel, self).__init__()

        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.1)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768, 512)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(512, 2)

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    # define the forward pass
    def forward(self, sent_id, mask):
        # pass the inputs to the model
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)

        x = self.fc1(cls_hs)

        x = self.relu(x)

        x = self.dropout(x)

        # output layer
        x = self.fc2(x)

        # apply softmax activation
        x = self.softmax(x)

        return x


class BertSpamDetectorTrainer:
    def __init__(self):
        if not os.path.exists(args.model_output_dir):
            os.makedirs(args.model_output_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.df = pd.read_csv(args.training_data_file)

        self.train_text, \
            self.temp_text, \
            self.train_labels, \
            self.temp_labels = train_test_split(self.df['text'], self.df['label'],
                                                random_state=2018,
                                                test_size=0.3,
                                                stratify=self.df[
                                                    'label'])

        self.val_text, \
            self.test_text, \
            self.val_labels, \
            self.test_labels = train_test_split(self.temp_text,
                                                self.temp_labels,
                                                random_state=2018,
                                                test_size=0.5,
                                                stratify=self.temp_labels)

        self.bert = AutoModel.from_pretrained('bert-base-uncased',
                                              cache_dir=args.transformer_cache_dir,
                                              revision=args.bert_revision)

        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased',
                                                           cache_dir=args.transformer_cache_dir,
                                                           revision=args.bert_revision)

        self.max_seq_len = 25

        self.tokens_train = self.tokenizer.batch_encode_plus(
            self.train_text.tolist(),
            max_length=self.max_seq_len,
            padding=True,
            truncation=True,
            return_token_type_ids=False
        )

        self.tokens_val = self.tokenizer.batch_encode_plus(
            self.val_text.tolist(),
            max_length=self.max_seq_len,
            padding=True,
            truncation=True,
            return_token_type_ids=False
        )

        self.tokens_test = self.tokenizer.batch_encode_plus(
            self.test_text.tolist(),
            max_length=self.max_seq_len,
            padding=True,
            truncation=True,
            return_token_type_ids=False
        )

        self.train_seq = torch.tensor(self.tokens_train['input_ids'])
        self.train_mask = torch.tensor(self.tokens_train['attention_mask'])
        self.train_y = torch.tensor(self.train_labels.tolist())

        self.val_seq = torch.tensor(self.tokens_val['input_ids'])
        self.val_mask = torch.tensor(self.tokens_val['attention_mask'])
        self.val_y = torch.tensor(self.val_labels.tolist())

        self.test_seq = torch.tensor(self.tokens_test['input_ids'])
        self.test_mask = torch.tensor(self.tokens_test['attention_mask'])
        self.test_y = torch.tensor(self.test_labels.tolist())

        self.batch_size = 32

        self.train_data = TensorDataset(self.train_seq, self.train_mask, self.train_y)

        self.train_sampler = RandomSampler(self.train_data)

        self.train_dataloader = DataLoader(self.train_data, sampler=self.train_sampler, batch_size=self.batch_size)

        self.val_data = TensorDataset(self.val_seq, self.val_mask, self.val_y)

        self.val_sampler = SequentialSampler(self.val_data)

        self.val_dataloader = DataLoader(self.val_data, sampler=self.val_sampler, batch_size=self.batch_size)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.model = BertSpamDetectorModel(self.bert)

        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        self.class_wts = compute_class_weight('balanced', classes=np.unique(self.train_labels), y=self.train_labels)

        self.weights = torch.tensor(self.class_wts, dtype=torch.float)
        self.weights = self.weights.to(self.device)

        self.cross_entropy = nn.NLLLoss(weight=self.weights)

        self.epochs = 10

    def train(self):
        self.model.train()

        total_loss, total_accuracy = 0, 0

        total_preds = []

        for step, batch in enumerate(self.train_dataloader):
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.train_dataloader)))

            batch = [r.to(self.device) for r in batch]

            sent_id, mask, labels = batch

            self.model.zero_grad()

            preds = self.model(sent_id, mask)
            loss = self.cross_entropy(preds, labels)
            total_loss = total_loss + loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

        avg_loss = total_loss / len(self.train_dataloader)

        total_preds = np.concatenate(total_preds, axis=0)

        return avg_loss, total_preds

    def evaluate(self):
        print("\nEvaluating...")

        self.model.eval()

        total_loss, total_accuracy = 0, 0
        total_preds = []

        for step, batch in enumerate(self.val_dataloader):
            if step % 50 == 0 and not step == 0:
                print('Batch {:>5,}  of  {:>5,}.'.format(step, len(self.val_dataloader)))

            batch = [t.to(self.device) for t in batch]

            sent_id, mask, labels = batch

            with torch.no_grad():
                preds = self.model(sent_id, mask)
                loss = self.cross_entropy(preds, labels)
                total_loss = total_loss + loss.item()

                preds = preds.detach().cpu().numpy()
                total_preds.append(preds)

        avg_loss = total_loss / len(self.val_dataloader)

        total_preds = np.concatenate(total_preds, axis=0)

        return avg_loss, total_preds

    def run(self):
        best_valid_loss = float('inf')

        train_losses = []
        valid_losses = []

        # for each epoch
        for epoch in range(self.epochs):

            print('\n Epoch {:} / {:}'.format(epoch + 1, self.epochs))

            train_loss, _ = self.train()
            valid_loss, _ = self.evaluate()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), '{}/pytorch_model.bin'.format(args.model_output_dir))

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            print(f'\nTraining Loss: {train_loss:.3f}')
            print(f'Validation Loss: {valid_loss:.3f}')


if __name__ == "__main__":
    trainer = BertSpamDetectorTrainer()
    trainer.run()
