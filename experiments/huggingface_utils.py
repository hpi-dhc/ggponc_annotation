from datasets import load_dataset, load_metric, Sequence, ClassLabel, DatasetDict
import transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments, pipeline, DataCollatorForTokenClassification
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import error_analysis
import pandas as pd

metric = load_metric("seqeval")

class LabelAligner():
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize_and_align_labels(self, examples, label_all_tokens=True):
        tokenized_inputs = self.tokenizer(examples["tokens"], 
                                          truncation=True, 
                                          is_split_into_words=True, 
                                          return_offsets_mapping=True,
                                          return_special_tokens_mask=True)

        labels = []
        for i, label in enumerate(examples["_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

def load_custom_dataset(train, dev, test, tag_strings=None):
    tags = []
    dataset = load_dataset("json", data_files={'train' : str(train), 'dev' : str(dev), 'test' : str(test)})
    features = dataset["train"].features

    tags.append("O")
    for tag in tag_strings:
        tags.append("B-" + tag)
        tags.append("I-" + tag)
    tag2idx = defaultdict(int)
    tag2idx.update({t: i for i, t in enumerate(tags)})
    dataset = dataset.map(lambda e: {"_tags" : [tag2idx[t] for t in e["tags"]]})
    features["_tags"] = Sequence(ClassLabel(num_classes=len(tags), names=(tags)))
        
    dataset = dataset.cast(features)
        
    return dataset, tags

def compute_metrics(label_list, entity_level_metrics):
    def _compute_metrics(p):
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
        
        if entity_level_metrics:
            final_results = {}
            # Unpack nested dictionaries
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    return _compute_metrics

def eval_on_test_set(test_ds, trainer, tokenizer, prefix): 
    pred = trainer.predict(test_ds)
        
    pred_labels = pred.predictions.argmax(axis=2)
    
    ner_stats = []
    
    for i, z in tqdm(enumerate(zip(test_ds, pred_labels, test_ds['special_tokens_mask']))):
        sentence, sentence_pred, special_tokens_mask = z
        ea = error_analysis.ner_error_analyis(
            sentence, sentence["labels"], sentence_pred, special_tokens_mask, tokenizer, trainer.model.config.id2label, skip_subwords=True)
        ea = [dict({'sentence_id' : i}, **e) for e in ea]
        ner_stats += ea
    
    stats_df = pd.DataFrame(ner_stats)
    
    error_count = stats_df.groupby('category').size()
    stats_df.to_csv('error_analysis.csv')
    
    error_count = error_count.to_dict()
    return {prefix + '/' + k.replace('test_', '') : v for k, v in dict(error_count.items(), **pred.metrics).items()}