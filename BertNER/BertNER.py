import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras_preprocessing.sequence import pad_sequences
from seqeval.metrics import f1_score
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from tqdm import tqdm, trange


class SentenceGetter(object):

    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["words"].values.tolist(),
                                                     s["tags"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_index").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


if __name__ == "__main__":
    MAX_LEN = 50
    bs = 64
    epochs = 3
    max_grad_norm = 1.0
    save_path = "BertNER/BertNER.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dframe = pd.read_csv("csv/train.csv")
    val_dframe = pd.read_csv("csv/val.csv")

    train_getter = SentenceGetter(train_dframe)
    val_getter = SentenceGetter(val_dframe)

    train_sentences = [" ".join([str(s[0]) for s in sent]) for sent in train_getter.sentences]
    val_sentences = [" ".join([str(s[0]) for s in sent]) for sent in val_getter.sentences]
    print('train_sentences[0]: ', train_sentences[0])

    train_labels = [[s[1] for s in sent] for sent in train_getter.sentences]
    val_labels = [[s[1] for s in sent] for sent in val_getter.sentences]
    print('train_labels[0]: ', train_labels[0])

    tags_vals = list(set(train_dframe["tags"].values))
    tag2idx = {t: i for i, t in enumerate(tags_vals)}

    train_tokenized_texts = [tokenizer.tokenize(sent) for sent in train_sentences]
    val_tokenized_texts = [tokenizer.tokenize(sent) for sent in val_sentences]
    print('train_tokenized_texts[0]: ', train_tokenized_texts[0])

    # get input id
    train_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in train_tokenized_texts],
                                    maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    val_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in val_tokenized_texts],
                                  maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # get tags
    train_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in train_labels],
                               maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                               dtype="long", truncating="post")
    val_tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in val_labels],
                             maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                             dtype="long", truncating="post")

    # get masks
    train_attention_masks = [[float(i > 0) for i in ii] for ii in train_input_ids]
    val_attention_masks = [[float(i > 0) for i in ii] for ii in val_input_ids]

    # to tensor
    train_input_ids = torch.tensor(train_input_ids)
    train_tags = torch.tensor(train_tags)
    train_attention_masks = torch.tensor(train_attention_masks)

    val_input_ids = torch.tensor(val_input_ids)
    val_tags = torch.tensor(val_tags)
    val_attention_masks = torch.tensor(val_attention_masks)

    # get dataloader
    train_data = TensorDataset(train_input_ids, train_attention_masks, train_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

    valid_data = TensorDataset(val_input_ids, val_attention_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

    # load model from hugging face
    model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(tag2idx)).to(device)

    # get parameter need to fine-tune
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

    # train !!

    for _ in trange(epochs, desc="Epoch"):
        # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            # forward pass
            loss = model(b_input_ids, token_type_ids=None,
                         attention_mask=b_input_mask, labels=b_labels)
            # backward pass
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            model.zero_grad()
        # print train loss per epoch
        print("Train loss: {}".format(tr_loss / nb_tr_steps))
        # VALIDATION on validation set
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                      attention_mask=b_input_mask, labels=b_labels)
                logits = model(b_input_ids, token_type_ids=None,
                               attention_mask=b_input_mask)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.append(label_ids)

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss / nb_eval_steps
        print("Validation loss: {}".format(eval_loss))
        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
        pred_tags = [tags_vals[p_i] for p in predictions for p_i in p]
        valid_tags = [tags_vals[l_ii] for l in true_labels for l_i in l for l_ii in l_i]

    # save model
    torch.save(model.state_dict(), save_path)

    # model.load_state_dict(torch.load(save_path))
    model.eval()
    predictions = []
    true_labels = []
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_mask)

        logits = logits.detach().cpu().numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        label_ids = b_labels.to('cpu').numpy()
        true_labels.append(label_ids)
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

    pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]
    valid_tags = [[tags_vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l]
    print("Validation loss: {}".format(eval_loss / nb_eval_steps))
    print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    print("Validation F1-Score: {}".format(flat_accuracy(pred_tags, valid_tags)))




