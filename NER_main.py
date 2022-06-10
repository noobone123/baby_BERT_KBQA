import argparse
from lib2to3.pgen2 import token
import logging
import codecs
import os
import random
from re import M
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

        # input_ids 的长度为 example.text 的长度+2，具体是在句子开头添加了 [CLS]，在句子结尾添加了 [SEP]
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


def statistical_real_sentences(input_ids:torch.Tensor,mask:torch.Tensor,predict:list)-> list:
    # shape (batch_size,max_len)
    assert input_ids.shape == mask.shape
    # batch_size
    assert input_ids.shape[0] == len(predict)

    # 第0位是[CLS] 最后一位是<pad> 或者 [SEP]
    new_ids = input_ids[:,1:-1]
    new_mask = mask[:,2:]

    real_ids = []
    for i in range(new_ids.shape[0]):
        seq_len = new_mask[i].sum()
        assert seq_len == len(predict[i])
        real_ids.append(new_ids[i][:seq_len].tolist())
    return real_ids

def flatten(inputs:list) -> list:
    result = []
    [result.extend(line) for line in inputs]
    return result

def evaluate(model, eval_dataset, device):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler,
                                 batch_size= ModelConfig["eval_batch_size"])

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", ModelConfig["eval_batch_size"])

    loss = []
    real_token_label = []
    pred_token_label = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':batch[0],
                      'attention_mask':batch[1],
                      'token_type_ids':batch[2],
                      'tags':batch[3],
                      'decode':True,
                      'reduction':'none'
            }
            outputs = model(**inputs)
            # temp_eval_loss shape: (batch_size)
            # temp_pred : list[list[int]] 长度不齐
            temp_eval_loss, temp_pred = outputs[0], outputs[1]

            loss.extend(temp_eval_loss.tolist())
            pred_token_label.extend(temp_pred)
            real_token_label.extend(statistical_real_sentences(batch[3],batch[1],temp_pred))

    loss = np.array(loss).mean()
    real_token_label = np.array(flatten(real_token_label))
    pred_token_label = np.array(flatten(pred_token_label))
    assert real_token_label.shape == pred_token_label.shape
    ret = classification_report(y_true = real_token_label,y_pred = pred_token_label,output_dict = True)
    model.train()
    return ret

def evaluate_and_save_model(model, eval_dataset, epoch, global_step, best_f1, device):
    ret = evaluate(model, eval_dataset, device)

    precision_b = ret['1']['precision']
    recall_b = ret['1']['recall']
    f1_b = ret['1']['f1-score']
    support_b = ret['1']['support']

    precision_i = ret['2']['precision']
    recall_i = ret['2']['recall']
    f1_i = ret['2']['f1-score']
    support_i = ret['2']['support']

    weight_b = support_b / (support_b + support_i)
    weight_i = 1 - weight_b

    avg_precision = precision_b * weight_b + precision_i * weight_i
    avg_recall = recall_b * weight_b + recall_i * weight_i
    avg_f1 = f1_b * weight_b + f1_i * weight_i

    all_avg_precision = ret['macro avg']['precision']
    all_avg_recall = ret['macro avg']['recall']
    all_avg_f1 = ret['macro avg']['f1-score']

    logger.info("Evaluating EPOCH = [%d/%d] global_step = %d", epoch+1, ModelConfig["num_train_epochs"], global_step)
    logger.info("B-LOC precision = %f recall = %f  f1 = %f support = %d", precision_b, recall_b, f1_b,
                support_b)
    logger.info("I-LOC precision = %f recall = %f  f1 = %f support = %d", precision_i, recall_i, f1_i,
                support_i)

    logger.info("attention AVG:precision = %f recall = %f  f1 = %f ", avg_precision, avg_recall,
                avg_f1)
    logger.info("all AVG:precision = %f recall = %f  f1 = %f ", all_avg_precision, all_avg_recall,
                all_avg_f1)

    if avg_f1 > best_f1:
        best_f1 = avg_f1

        eval_output_dirs = ModelConfig["output_dir"]
        if not os.path.exists(eval_output_dirs):
            os.makedirs(eval_output_dirs)
        
        model_file = os.path.join(ModelConfig["output_dir"], "best_ner.bin")
        if os.path.exists(model_file):
            os.remove(model_file)

        torch.save(model.state_dict(), model_file)
        logging.info("save the best model %s,avg_f1= %f", os.path.join(ModelConfig["output_dir"], "best_bert.bin"),
                     best_f1)

    # 返回出去，用于更新外面的 最佳值
    return best_f1


def train(train_dataset, eval_dataset, model, device):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=ModelConfig["train_batch_size"])

    t_total = len(train_dataloader)
    no_decay = ['bias', 'LayerNorm.weight','transitions']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': ModelConfig["weight_decay"]},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr = ModelConfig["learning_rate"], eps = ModelConfig["adam_epsilon"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=ModelConfig["warmup_steps"], num_training_steps=t_total)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", ModelConfig["num_train_epochs"])
    logger.info("  Gradient Accumulation steps = %d", ModelConfig["gradient_accumulation_steps"])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(ModelConfig["num_train_epochs"]), desc="Epoch")

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    set_seed(ModelConfig["seed"])

    best_f1 = 0.
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step,batch in enumerate(epoch_iterator):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':batch[0],
                      'attention_mask':batch[1],
                      'token_type_ids':batch[2],
                      'tags':batch[3],
                      'decode':True
            }
            outputs = model(**inputs)
            loss, pre_tag = outputs[0], outputs[1]

            if ModelConfig["gradient_accumulation_steps"] > 1:
                loss = loss / ModelConfig["gradient_accumulation_steps"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), ModelConfig["max_grad_norm"])
            logging_loss += loss.item()
            tr_loss += loss.item()

            if 0 == (step + 1) % ModelConfig["gradient_accumulation_steps"]:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                logger.info("EPOCH = [%d/%d] global_step = %d   loss = %f",_+1, ModelConfig["num_train_epochs"], global_step,
                            logging_loss)
                logging_loss = 0.0

                # if (global_step < 100 and global_step % 10 == 0) or (global_step % 50 == 0):
                # 每 相隔 100步，评估一次
                if global_step % 100 == 0:
                    best_f1 = evaluate_and_save_model(model, eval_dataset, _ , global_step, best_f1, device)

    # 最后循环结束 再评估一次
    best_f1 = evaluate_and_save_model(model, eval_dataset, _ , global_step, best_f1, device)


def test(test_dataset, device):
    model_path = os.path.join(ModelConfig["output_dir"], "best_ner.bin")
    model = BERT_CRF( config_file_path = ModelConfig["model_config"],
                        num_tags = len(get_labels()),
                        batch_first = True )
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    sampler = RandomSampler(test_dataset)
    data_loader = DataLoader(test_dataset, sampler=sampler, batch_size=256)
    loss = []
    real_token_label = []
    pred_token_label = []

    for batch in tqdm(data_loader, desc="test"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'tags': batch[3],
                    'decode': True,
                    'reduction': 'none'
                    }
            outputs = model(**inputs)
            # temp_eval_loss shape: (batch_size)
            # temp_pred : list[list[int]] 长度不齐
            temp_eval_loss, temp_pred = outputs[0], outputs[1]

            loss.extend(temp_eval_loss.tolist())
            pred_token_label.extend(temp_pred)
            real_token_label.extend(statistical_real_sentences(batch[3], batch[1], temp_pred))

    loss = np.array(loss).mean()
    real_token_label = np.array(flatten(real_token_label))
    pred_token_label = np.array(flatten(pred_token_label))
    assert real_token_label.shape == pred_token_label.shape
    ret = classification_report(y_true=real_token_label, y_pred=pred_token_label, digits = 6,output_dict=False)

    print(ret)

    

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action='store_true', help="是否进行训练")
    parser.add_argument("--do_test", action='store_true', help="是否进行训练")
    args = parser.parse_args()

    if args.do_train:
        train(train_dataset, eval_dataset, model, device)
    if args.do_test:
        test(test_dataset, device)


if __name__ == "__main__":
    main()