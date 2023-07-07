import json
import pandas as pd
from datasets import load_metric, Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import numpy as np
import torch
import time


def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def load_BioRED(mode):
    if mode == 'train':
        data = load_json('../dataset/JSON1/train1.json')
    elif mode == 'val':
        data = load_json('../dataset/JSON1/val1.json')

    label_list = ['B-CellLine',
                  'B-ChemicalEntity',
                  'B-DiseaseOrPhenotypicFeature',
                  'B-GeneOrGeneProduct',
                  'B-OrganismTaxon',
                  'B-SequenceVariant',
                  'I-CellLine',
                  'I-ChemicalEntity',
                  'I-DiseaseOrPhenotypicFeature',
                  'I-GeneOrGeneProduct',
                  'I-OrganismTaxon',
                  'I-SequenceVariant',
                  'O']
    return data['train'], data['labels'], label_list


def tokenize_and_align_labels(examples):
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["words"]), truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            elif label[word_idx] == '0':
                label_ids.append(0)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_encoding_dict[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label_encoding_dict[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


if __name__ == '__main__':
    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ############################################################
    model_checkpoint = "d4data/biomedical-ner-all"
    ############################################################
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    train_words, train_tags, label_list = load_BioRED('train')
    val_words, val_tags, _ = load_BioRED('val')
    label_encoding_dict = {tag: i for i, tag in enumerate(label_list)}

    train_df = pd.DataFrame({'words': train_words, 'tags': train_tags})
    val_df = pd.DataFrame({'words': val_words, 'tags': val_tags})

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
    val_tokenized_datasets = val_dataset.map(tokenize_and_align_labels, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(label_list), ignore_mismatched_sizes=True)

    args = TrainingArguments(
        f"biomedical-ner-all",
        evaluation_strategy="epoch",
        learning_rate=1e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        weight_decay=0.00001,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)
    metric = load_metric("seqeval")

    trainer = Trainer(
        model,
        args,
        train_dataset=train_tokenized_datasets,
        eval_dataset=val_tokenized_datasets,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    start = time.time()
    trainer.train()
    print('training time: ', time.time()-start)

    result = trainer.evaluate()
    print(result)
