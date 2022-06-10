import torch
import torch.nn as nn
import os
from typing import List, Optional
from transformers import BertForTokenClassification, BertTokenizer, BertConfig
from Model.CRF import CRF

class BERT_CRF(nn.Module):
    def __init__(self,
                config_file_path:str,
                model_name: Optional[str] = None,
                num_tags: int = 2,
                batch_first: bool = True) -> None:

        self.batch_first = batch_first

        # 验证 BERT 模型的配置文件和预训练好的参数文件是否存在
        if not os.path.exists(config_file_path):
            raise ValueError("Can't find config file {}".format(config_file_path))
        else:
            self.config_file_path = config_file_path

        if model_name is not None:
            if not os.path.exists(model_name):
                raise ValueError("Can't find pretrained model {}".format(model_name))
            else:
                self.model_name = model_name
        else:
            self.model_name = None

        super().__init__()

        self.bert_config = BertConfig.from_pretrained(self.config_file_path)
        self.bert_config.num_labels = num_tags
        self.model_kwargs = {'config': self.bert_config}

        if self.model_name is not None:
            self.bert_model = BertForTokenClassification.from_pretrained(self.model_name, **self.model_kwargs)
        else:
            self.bert_model = BertForTokenClassification(self.bert_config)
        
        self.crf_model = CRF(num_tags=num_tags, batch_first=batch_first)

    
    def forward(self,
                input_ids: torch.Tensor,
                tags: torch.Tensor = None,
                attention_mask:Optional[torch.ByteTensor] = None,
                token_type_ids=torch.Tensor,
                decode:bool = True,
                reduction: str = "mean") -> List:
        
        emissions = self.bert_model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)[0]

        new_emissions = emissions[:,1:-1]
        new_mask = attention_mask[:,2:].bool()

        if tags is None:
            loss = None
            pass
        else:
            new_tags = tags[:, 1:-1]
            loss = self.crf_model(emissions=new_emissions, tags=new_tags, mask=new_mask, reduction=reduction)
        
        if decode:
            tag_list = self.crf_model.decode(emissions = new_emissions,mask = new_mask)
            return [loss, tag_list]

        return [loss]
