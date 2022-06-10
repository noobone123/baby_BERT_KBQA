import argparse
from lib2to3.pgen2 import token
import logging
import codecs
import os
import random
import numpy as np
import torch
from tqdm import tqdm, trange

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertForSequenceClassification,BertTokenizer,BertConfig
from transformers.data.processors.utils import DataProcessor, InputExample
from Model.BERT_CRF import BERT_CRF
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from global_config import *

logger = logging.getLogger(__name__)

CRF_LABELS = ["O", "B-EN", "I-EN"]

def get_labels():
    return CRF_LABELS

def create_examples(path):
    lines = []
    max_len = 0
    with codecs.open(path, "r", encoding='utf-8') as f:
        word_list = []
        label_list = []
        for line in f:
            tokens = line.strip().split(" ")
            if len(tokens) == 2:
                word = tokens[0]
                label = tokens[1]
                word_list.append(word) 
                label_list.append(label)
            elif len(tokens) == 1 and tokens[0] == "":
                if len(label_list) > max_len:
                    max_len = len(label_list)

                lines.append((word_list, label_list))
                word_list = []
                label_list = []
    
    examples = []

    class CrfInputExample(object):
        def __init__(self, guid, text, label=None):
            self.guid = guid
            self.text = text
            self.label = label

    for i,(sentence, label) in enumerate(lines):
        examples.append(
            CrfInputExample(guid = i, text = " ".join(sentence), label = label)
        )
    return examples

def crf_convert_features(examples, tokenizer:BertTokenizer,
                        max_length=512,
                        label_list=None,
                        pad_token=0,
                        pad_token_segment_id = 0,
                        mask_padding_with_zero = True):
    label_map = {label:i for i,label in enumerate(label_list)}
    features = []
    for (index, example) in enumerate(examples):
        inputs = tokenizer.encode_plus(
            example.text,
            add_special_tokens = True,
            max_length = max_length,
            truncation = True
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        attention_mask = [1 if mask_padding_with_zero else 0 ] * len(input_ids)

        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        # 第一个和第二个[0] 加的是[CLS]和[SEP]的位置,  [0]*padding_length是[pad] ，把这些都暂时算作"O"，后面用mask 来消除这些，不会影响
        labels_ids = [0] + [label_map[l] for l in example.label] + [0] + [0]*padding_length
        
        assert len(input_ids) == max_length
        assert len(attention_mask) == max_length
        assert len(token_type_ids) == max_length
        assert len(labels_ids) == max_length

        class CrfInputFeatures():
            def __init__(self, input_ids, attention_mask, token_type_ids, label):
                self.input_ids = input_ids
                self.attention_mask = attention_mask
                self.token_type_ids = token_type_ids
                self.label = label
        
        features.append(
            CrfInputFeatures(input_ids, attention_mask, token_type_ids, labels_ids)
        )

    return features

def construct_datasets(tokenizer, dataset_type_list):
    if len(dataset_type_list) != 3:
        raise ValueError("data set type error!")

    all_datasets = []

    for dataset_type in dataset_type_list:
        label_list = get_labels()
        file_full_path = os.path.join(ModelConfig["data_dir"], dataset_type + ".txt")
        examples = create_examples(file_full_path)

        features = crf_convert_features(examples = examples, tokenizer = tokenizer, max_length = ModelConfig["max_seq_length"], label_list = label_list)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype = torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype = torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype = torch.long)
        all_label = torch.tensor([f.label for f in features], dtype = torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_label)
        all_datasets.append(dataset)

    return all_datasets

def train(train_dataset, eval_dataset, model):
    pass

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Current device is: {}".format(device))

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    
    tokenizer_inputs = ()
    tokenizer_kwards = {'do_lower_case': False,
                        'max_len': ModelConfig["max_seq_length"],
                        'vocab_file': ModelConfig["vob_file"]}

    tokenizer = BertTokenizer(*tokenizer_inputs, **tokenizer_kwards)

    model = BERT_CRF( config_file_path = ModelConfig["model_config"],
                        model_name = ModelConfig["pre_train_model"],
                        num_tags = len(get_labels()),
                        batch_first = True )
    model = model.to(device)
    
    train_dataset, test_dataset, eval_dataset = construct_datasets(tokenizer, ["train_data", "test_data", "dev_data"])
    logger.info("Construct datasets successfully!")

    train(train_dataset, eval_dataset, model)


if __name__ == "__main__":
    main()