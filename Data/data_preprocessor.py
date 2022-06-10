"""
    data preprocessor:
    
"""

from curses import raw
from html import entities
from inspect import Attribute
import pandas as pd
import os
import sys
import re
import random

sys.path.append("..")
from global_config import *

# get origin data files
ori_data_dir = OriDataConfig["ori_data_dir_path"]
ori_train_file = OriDataConfig["ori_training_data"]
ori_test_file = OriDataConfig["ori_testing_data"]

train_file_full_path = os.path.join(ori_data_dir, ori_train_file)
test_file_full_path = os.path.join(ori_data_dir, ori_test_file)

def check_file_exists(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        return

"""
    将数据集进行切分：
    1. 对于原始的 training data，去掉空行后保持不变
    2. 对于原始的 testing data，切分前一半为测试集，后一半为开发集
"""
def split_data(file_path):
    res = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            res.append(line)
        f.close()

    # 断言：样本的完整性，一个样本由 4 行构成
    assert len(res) % 4 == 0

    if "training" in file_path:
        target_file_path = os.path.join(ori_data_dir, OriDataConfig["splited_training_data"])

        # 检查文件是否存在
        check_file_exists(target_file_path)

        with open(target_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(res))
        f.close()

    elif "testing" in file_path:
        target_file_path_1 = os.path.join(ori_data_dir, OriDataConfig["splited_testing_data"])
        target_file_path_2 = os.path.join(ori_data_dir, OriDataConfig["splited_dev_data"])

        # 检查文件是否存在
        check_file_exists(target_file_path_1)
        # 检查文件是否存在
        check_file_exists(target_file_path_2)

        sample_cnt = len(res) // 4
        split_line_no = (sample_cnt // 2) * 4

        # 前一半数据作为测试集
        with open(target_file_path_1, "w", encoding="utf-8") as f:
            f.write("\n".join(res[:split_line_no]))
        f.close()

        # 后一半数据作为开发集
        with open(target_file_path_2, "w", encoding="utf-8") as f:
            f.write("\n".join(res[split_line_no:]))
        f.close()
    
    print("Data Split done!")
    
    return

"""
    构建基础的用于训练 NER 的数据：
    1. 对 question sentence 中的实体进行标注
    2. 将 question,triple,answer 构建成 csv 文件
"""
def construct_ner_dataset():
    ori_data_dir_path = OriDataConfig["ori_data_dir_path"]
    splited_training_data = OriDataConfig["splited_training_data"]
    splited_testing_data = OriDataConfig["splited_testing_data"]
    splited_dev_data = OriDataConfig["splited_dev_data"]

    raw_data_files = [
                        os.path.join(ori_data_dir_path, splited_training_data),
                        os.path.join(ori_data_dir_path, splited_testing_data),
                        os.path.join(ori_data_dir_path, splited_dev_data)
                    ]
    
    for file in raw_data_files:
        # 判断文件存在
        assert os.path.exists(file)
        with open(file, "r", encoding="utf-8") as f:
            question_seq_list = []
            question_tag_list = []
            q_t_a_list = []

            question = ""
            triple = ""
            answer = ""
            line_no = 0

            for line in f:
                line_no += 1
                if line.startswith(OriDataConfig["question_str"]):
                    question = line.strip()
                elif line.startswith(OriDataConfig["triple_str"]):
                    triple = line.strip()
                elif line.startswith(OriDataConfig["answer_str"]):
                    answer = line.strip()
                elif line.startswith(OriDataConfig["split_str"]):
                    assert line_no % 4 == 0
                    entity = triple.split("|||")[0].split(">")[1].strip()
                    question = question.split(">")[1].replace(" ","").strip()

                    # 标记在问句中的实体
                    # 对于问句中的实体：开头的汉字标记为 B-EN，中间和结尾的汉字标记为 I-EN
                    if entity in question:
                        question_list = list(question)
                        question_seq_list.extend(question_list)
                        question_seq_list.extend([" "])
                        tag_list = ["O" for i in range(len(question_list))]
                        tag_start_index = question.find(entity)
                        for i in range(tag_start_index, tag_start_index + len(entity)):
                            if tag_start_index == i:
                                tag_list[i] = OriDataConfig["entity_start_tag"]
                            else:
                                tag_list[i] = OriDataConfig["entity_inner_tag"]

                        question_tag_list.extend(tag_list)
                        question_tag_list.extend([" "])
                    else:
                        pass

                    q_t_a_list.append([question, triple, answer])
        
        seq_result = [str(q) + " " + str(tag) for q, tag in zip(question_seq_list, question_tag_list)]

        ner_data_dir = OriDataConfig["ner_data_dir_path"]
        if not os.path.exists(ner_data_dir):
            os.mkdir(ner_data_dir)
        
        file_name = file.strip("NLPCC2016KBQA/")
        target_file_path_1 = os.path.join(ner_data_dir, file_name + ".txt")
        target_file_path_2 = os.path.join(ner_data_dir, file_name + ".csv")

        check_file_exists(target_file_path_1)
        check_file_exists(target_file_path_2)

        with open(target_file_path_1, "w", encoding="utf-8") as f:
            f.write("\n".join(seq_result))
        
        df = pd.DataFrame(q_t_a_list, columns=["question", "triple", "answer"])
        df.to_csv(target_file_path_2, encoding="utf-8", index=False)


"""
    通过 NER data 中的数据，构造用来匹配句子相似度的训练集
"""
def construct_sim_dataset():
    ner_data_dir = OriDataConfig["ner_data_dir_path"]
    sim_data_dir = OriDataConfig["similarity_data_dir_path"]
    file_name_list = [
        OriDataConfig["splited_training_data"] + ".csv",
        OriDataConfig["splited_testing_data"] + ".csv",
        OriDataConfig["splited_dev_data"] + ".csv"
    ]

    pattern = re.compile("^-+")

    for file_name in file_name_list:
        file_full_path = os.path.join(ner_data_dir, file_name)
        assert os.path.exists(file_full_path)

        attribute_classify_sample = []
        df = pd.read_csv(file_full_path, encoding="utf-8")

        # 为 DataFrame 加入新的列 attribute
        df['attribute'] = df['triple'].apply(lambda x: x.split("|||")[1].strip())
        attribute_list = df['attribute'].tolist()
        attribute_list = list(set(attribute_list))
        attribute_list = [a.strip().replace(" ","") for a in attribute_list]
        attribute_list = [re.sub(pattern, "", a) for a in attribute_list]
        attribute_list = list(set(attribute_list))

        for row in df.index:
            question, attribute = df.loc[row][['question', 'attribute']]
            question.strip().replace(" ","")
            question = re.sub(pattern, "", question)
            attribute.strip().replace(" ","")
            attribute = re.sub(pattern, "", attribute)

            # 对于每个 question 和对应的 attribute，
            # 从样本中收集 5 个不对应的 attribute 作为训练样本
            neg_att_list = []
            while True:
                neg_att_list = random.sample(attribute_list, 5)
                if attribute not in neg_att_list:
                    break

            attribute_classify_sample.append([question, attribute, "1"])
            neg_att_sample = [[question, neg_att, "0"] for neg_att in neg_att_list]
            attribute_classify_sample.extend(neg_att_sample)
        
        seq_result = [str(lineno) + '\t' + '\t'.join(line) for (lineno, line) in enumerate(attribute_classify_sample)]

        if not os.path.exists(sim_data_dir):
            os.makedirs(sim_data_dir)
        
        new_file_name = file_name.split(".")[0] + ".txt"
        new_file_full_path = os.path.join(sim_data_dir, new_file_name)
        
        check_file_exists(new_file_full_path)

        with open(new_file_full_path, "w", encoding="utf-8") as f:
            f.write("\n".join(seq_result))

"""
    计算 SIMdata 中 question 和 attribute 的最大字符数
"""
def cal_seq_maxlen():
    sim_data_dir = OriDataConfig["similarity_data_dir_path"]
    file_name_list = [
        OriDataConfig["splited_training_data"] + ".txt",
        OriDataConfig["splited_testing_data"] + ".txt",
        OriDataConfig["splited_dev_data"] + ".txt"
    ]

    for file in file_name_list:
        file_full_path = os.path.join(sim_data_dir, file)
        max_len = 0
        with open(file_full_path, "r", encoding="utf-8") as f:
            for line in f:
                line_list = line.split("\t")
                question = list(line_list[1])
                attribute = list(line_list[2])
                sum_len = len(question) + len(attribute)
                if sum_len > max_len:
                    max_len = sum_len
        
        """
        rain_data.txt : 64
        test_data.txt : 62
        dev_data.txt : 62
        """
        print(f"{file} : {max_len}")
    
def triple_clean():
    db_data_dir = OriDataConfig["database_data_dir_path"]
    ori_data_dir = OriDataConfig["ori_data_dir_path"]
    filenames = [
        OriDataConfig["ori_training_data"],
        OriDataConfig["ori_testing_data"]
    ]

    question_str = OriDataConfig["question_str"]
    triple_str = OriDataConfig["triple_str"]
    answer_str = OriDataConfig["answer_str"]
    split_str = OriDataConfig["split_str"]

    if not os.path.exists(db_data_dir):
        os.makedirs(db_data_dir)
    
    clean_triple_list = []
    for file in filenames:
        file_full_path = os.path.join(ori_data_dir, file)
        with open(file_full_path, "r", encoding="utf-8") as f:
            question, triple, answer = "", "", ""
            for line in f:
                if question_str in line:
                    question = line.strip()
                elif triple_str in line:
                    triple = line.strip()
                elif split_str in line:
                    entities = triple.split("|||")[0].split(">")[1].strip()
                    question = question.split(">")[1].replace(" ","").strip()
                    if "".join(entities.split(" ")) in question:
                        clean_triple = triple.split(">")[1].replace("\t", "").replace(" ","").strip().split("|||")
                        clean_triple_list.append(clean_triple)

    df = pd.DataFrame(clean_triple_list, columns=["entity", "attribute", "answer"])
    print(df.info())

    target_csv_file = os.path.join(db_data_dir, "clean_triple.csv")
    check_file_exists(target_csv_file)
    df.to_csv(target_csv_file, encoding="utf-8", index=False)


if __name__=="__main__":
    # split_data(train_file_full_path)
    # split_data(test_file_full_path)
    # print("Data split down!")

    # construct_ner_dataset()
    # print("Constructing NER dataset down!")

    # construct_sim_dataset()
    # print("Constructing SIM dataset down!")

    # cal_seq_maxlen()

    triple_clean()