import os

OriDataConfig = {
    "ori_data_dir_path" : "NLPCC2016KBQA",
    "ori_training_data" : "nlpcc-iccpol-2016.kbqa.training-data",
    "ori_testing_data" : "nlpcc-iccpol-2016.kbqa.testing-data",
    "splited_training_data" : "train_data",
    "splited_testing_data" : "test_data",
    "splited_dev_data" : "dev_data",

    "ner_data_dir_path" : "NER_Data",
    "question_str" : "<question",
    "triple_str" : "<triple",
    "answer_str" : "<answer",
    "split_str" : "======================",

    "similarity_data_dir_path" : "SIM_Data",

    "database_data_dir_path" : "DB_Data",

    "entity_start_tag": "B-EN",
    "entity_inner_tag": "I-EN"
}

DataSetConfig = {

}

DataLoaderConfig = {

}

ModelConfig = {
    "data_dir" : "./Data/NER_Data", 
    "vob_file" : "./Model/PretrainedModel/vocab.txt",
    "model_config" : "./Model/PretrainedModel/config.json",
    "output_dir" : "./output",
    "pre_train_model" : "./Model/PretrainedModel/bert-base-chinese-model.bin",
    "max_seq_length" : 64,
    "train_batch_size" : 32,
    "eval_batch_size" : 256,
    "gradient_accumulation_steps" : 4,
    "learning_rate" : 5e-5,
    "weight_decay" : 0.0,
    "adam_epsilon" : 1e-8,
    "max_grad_norm" : 1.0,
    "num_train_epochs" : 15,
    "seed" : 1234,
    "warmup_steps" : 0
}
