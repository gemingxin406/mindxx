import os
import json
import torch
from modelscope.models.cv.self_supervised_depth_completion.criteria import loss_names
from sympy.core.random import sample
from torch.utils.data import Dataset

#设置tokenizers不并行加速，避免报错
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#先写datdaset类
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512):
        super().__init__()

        self.samples =self.load_data(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
#再实现dataset内定的方法
    def load_data(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            for line_num,line in enumerate(f):
                data=json.loads(line.strip())
                sample.append(data)
            return sample
#_len_
    def __len__(self):
        return len(self.samples)
#_getitem_
    def __getitem__(self, index):
        sample = self.samples[index]
        encoding=self.tokenizer(
            str(sample),
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        loss_mask =input_ids!=self.tokenizer.pad_token_id
        #自回归
        X=torch.tensor(input_ids[:-1],dtype=torch.long)
        Y=torch.tensor(input_ids[1:],dtype=torch.long)

        loss_mask=torch.tensor(loss_mask[:-1],dtype=torch.long)

        return X,Y,loss_mask