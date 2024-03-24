import os
import sys
import json
import torch
import random
random.seed(42)
from itertools import chain
from multiprocessing import Pool
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

sys.path.append('/root/tuning_space/Components/')
from Static import prompt_dict, st, si
    
    
def build_labels(input_ids):
    # 初始化一个新列表，用于存储结果
    result = [-100] * len(input_ids)
    # 遍历列表，查找满足条件的 answer
    inside_ast = False  # 标记是否在<|assistant|>和 <|user|>之间
    for i, item in enumerate(input_ids):
        if item == si["<|assistant|>"]:
            inside_ast = True
        elif item == si["<|user|>"]:
            inside_ast = False
            result[i] = item
        elif inside_ast:
            result[i] = item
    #result = result[1:] + [-100]
    #result = [-100] + result[:-1]
    return result

class conversation_dataset(Dataset):
    def __init__(self, data_list, tokenizer, truncate_length, query_key, answer_key, max_sample_list=None, num_workers=12):
        super().__init__()
        self.tokenizer = tokenizer
        self.truncate_length = truncate_length
        self.query_key = query_key
        self.answer_key = answer_key
        self.max_sample_list = max_sample_list
        self.examples = []  # 存储最终结果的对象

        # 读取文件
        combine_lines = []
        for data_path, max_sample in zip(data_list, max_sample_list):
            with open(data_path, 'r') as file:
                lines = file.readlines()
            if max_sample:
                lines = random.sample(lines, min(max_sample, len(lines)))
            combine_lines += lines
        random.shuffle(combine_lines)

        # 创建一个进程池
        with Pool(num_workers) as p:
            self.examples = p.map(self.process_line, combine_lines)

    def process_line(self, line):
        try:
            conversation=json.loads(line)['conversation'] 
            '''
            此处假设conversation的query和answer均不会太长，故删除了判断长度的步骤
            '''
            # 制作input_ids
            input_ids = [si["[gMASK]"], si['sop']]

            # 遍历双方对话，首端添加特殊token
            for sample in conversation:
                role=next(iter(sample))
                if  role in self.query_key:
                    input_ids += [si["<|user|>"], si['\n']]
                    input_ids += self.tokenizer.encode(sample[role], add_special_tokens=False)
                elif role in self.answer_key:
                    #input_ids += [si["[gMASK]"], si['sop']]
                    input_ids += [si["<|assistant|>"], si['\n']]
                    #input_ids += [si["eop"]]
                    input_ids += self.tokenizer.encode(sample[role], add_special_tokens=False)

            # 判断截断，添加终止生成符号，padding
            if len(input_ids) > self.truncate_length-1:
                input_ids = input_ids[:self.truncate_length-1]
                #input_ids += [si["eop"]]
            input_ids += [si["<|user|>"]]
            input_ids += [self.tokenizer.pad_token_id] * (self.truncate_length-len(input_ids))

            # 制作labels
            labels = build_labels(input_ids)
            try:
                labels[labels.index(si["<|user|>"])]=-100
                #labels = [-100 if item == si["<|user|>"] else item for item in labels]
            except:
                print('fault')
                pass

            # 制作attention_mask
            try:
                eop_position = input_ids.index(self.tokenizer.pad_token_id)
            except:
                eop_position = len(input_ids)
            attention_mask = [True] * eop_position
            attention_mask += [False] * (self.truncate_length - len(attention_mask))

            return {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': attention_mask,
            }
        except:
            pass

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]
    
    
def coll_fn(batch:list):
    '''
    dataset的整理函数，目的是将items tensor化，并且成batch
    '''
    input_labels = []
    labels = []
    attention_mask = []
    for sample in batch:
        try:
            # 实际上词表长度只有65024，所以int32就可以了 # attention_mask用bool就行 (我收回我的画，完全是玄学)
            input_labels.append(torch.tensor(sample['input_ids'], dtype=torch.long))
            labels.append(torch.tensor(sample['labels'], dtype=torch.long))
            attention_mask.append(torch.tensor(sample['attention_mask'], dtype=torch.float64)) #, dtype=torch.bool
        except:
            pass
    batch = {'input_ids':input_labels, 'labels':labels, 'attention_mask': attention_mask}
    batch = {name:torch.stack(item).cuda() for name,item in batch.items()} #对于一个元素不为基本元素的list，需要使用stack方法
    return batch