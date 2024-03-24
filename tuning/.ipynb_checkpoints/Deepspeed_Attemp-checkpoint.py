class CFG:
    model_path = '/root/autodl-tmp/weights/chatglm3-6b'
    output_dir = '/root/autodl-tmp/checkpoints/glm3'
    #output_dir = '/root/autodl-tmp/checkpoints/glm3-full_query_turbo3'
    
    num_train_epochs = 10
    max_train_steps = None
    
    batch_size = 4 #需要和micro_batch_size同步
    max_tokens = 192
    max_query = 64
    
    lr = 1e-5
    warm_up_steps = 1000
    
    #data_path = '/root/autodl-tmp/dataset/OESD-GG-zh_cn-1/single_query.jsonl'
    #data_path = '/root/autodl-tmp/dataset/OESD-GG-zh_cn-1/full_query.jsonl'
    #query_key = 'User'
    #answer_key = 'Assisstant'
    
    data_path = '/root/autodl-tmp/dataset/psychology-dataset/data/train.jsonl'
    query_key = 'question'
    answer_key = 'response_j'
    
    #data_path = '/root/autodl-tmp/dataset/zhihu_qa/zhihu_qa_5w.jsonl'
    #query_key = 'question'
    #answer_key = 'answer'
    
    
import torch
import deepspeed
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss
from torch.utils.data.distributed import DistributedSampler
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModel
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig

import sys
import json
import pandas as pd
from tqdm import tqdm

sys.path.append('/root/tuning_space/Components/')
import interact
import model_tools
from Static import prompt_dict, st, si


config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],  
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

#载入model
tokenizer = AutoTokenizer.from_pretrained(CFG.model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(CFG.model_path, trust_remote_code=True).half()#.cuda().half()#.float()

#施加peft lora
model_tools.model_profile(model)
print('conducting peft lora ---------------')
model = get_peft_model(model, config)
model_tools.model_profile(model)


class instruction_dataset(Dataset):
    def __init__(self, data_path:'str', tokenizer, truncate_length, max_query_length, query_key, answer_key):
        super().__init__()
        self.tokenizer = tokenizer
        self.examples = []
        
        with open(data_path, 'r') as file:
            for line in file:
                sample=json.loads(line)
                # input_ids的结构应该为：prompt_tokens, src_tokens, [gMASK], <sop>, tgt_tokens, <eop>, [PAD]... 
                # 或者简化一点，即为 query, [gMASK], <sop>, answer, <eop>, [PAD]... 
                # padding的目的是为了对齐各个instance，以组成batch（当然batch_size=1时其实没必要）
                # 总体的input_ids的长度不超过truncate_length，其中query的长度不超过max_query_length，同理可以计算出answer的最大长度
                max_answer_length = truncate_length - max_query_length - 3
                
                # 判断query的长度
                query = sample[query_key]
                query_ids = tokenizer.encode(query, add_special_tokens=False)
                if len(query_ids) > max_query_length:
                    query_ids = query_ids[:max_query_length]
                
                # 判断answer的长度
                answer = sample[answer_key]
                answer_ids = tokenizer.encode(answer, add_special_tokens=False)
                if len(answer) > max_answer_length:
                    answer_ids = answer_ids[:max_answer_length]
                    
                # 合并
                input_ids = query_ids + [si['[gMASK]']] + [si['sop']] + answer_ids + [si['eop']]
                pre_context_length = input_ids.index(si['sop'])
                end_answer_index = input_ids.index(si['eop'])
                
                # padding
                padding_length=truncate_length-len(input_ids)
                input_ids+=padding_length*[tokenizer.pad_token_id]
                
                # 制作labels；其中query部分，pad部分均不参与loss的计算 # 因为需要整体向左移动，所以要少填充一个
                labels = [-100] * (pre_context_length+1) + input_ids[pre_context_length+1: end_answer_index+1]
                labels = labels + [-100] * (truncate_length-len(labels))
                
                # 制作attention_mask
                eop_position = input_ids.index(si['eop'])+1
                attention_mask = [True]*eop_position
                attention_mask += [False]*(truncate_length-len(attention_mask))
                
                self.examples.append({
                    'query' : query,
                    'answer' : answer,
                    'input_ids' : input_ids,
                    'labels' : labels,
                    'attention_mask' : attention_mask,
                })
                
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        instance = self.examples[item]
        return instance
    

def coll_fn(batch:list):
    input_labels = []
    labels = []
    attention_mask = []
    for sample in batch:
        # 实际上词表长度只有65024，所以int32就可以了 # attention_mask用bool就行 (我收回我的画，完全是玄学)
        input_labels.append(torch.tensor(sample['input_ids'], dtype=torch.long))
        labels.append(torch.tensor(sample['labels'], dtype=torch.long))
        attention_mask.append(torch.tensor(sample['attention_mask'], dtype=torch.float64)) #, dtype=torch.bool
    batch = {'input_ids':input_labels, 'labels':labels, 'attention_mask': attention_mask}
    batch = {name:torch.stack(item).cuda() for name,item in batch.items()} #对于一个元素不为基本元素的list，需要使用stack方法
    return batch


#将所有组件置于deepspeed的监控下
print('将所有组件置于deepspeed的监控下')
engine, optimizer, training_dataloader, lr_schedule = deepspeed.initialize(model=model, config='ds_config.json')
print(engine)

# 创建分布式采样器
finetuning_instruction = instruction_dataset(CFG.data_path, tokenizer, CFG.max_tokens, CFG.max_query, CFG.query_key, CFG.answer_key)#[:300]
print('创建分布式采样器')
sampler = DistributedSampler(finetuning_instruction)
# 使用分布式采样器创建数据加载器
instruction_loader = DataLoader(finetuning_instruction, shuffle=False, batch_size=CFG.batch_size, sampler=sampler, collate_fn=coll_fn)
print('dataloader的长度：',len(instruction_loader))

'''
for item in instruction_loader:
    print(item['input_ids'])
    #print(item)
    break
'''

# 训练阶段
engine.train()

for epoch in range(CFG.num_train_epochs):
    sampler.set_epoch(epoch)
    for step, batch in tqdm(enumerate(instruction_loader)):
        # 前向传播 & 计算loss
        outputs = engine(**batch)
        loss = outputs.loss
        print('')
        print(loss)

        # 反向传播
        engine.backward(loss)

        # 优化器调度
        engine.step()
        
#模型的保存
model.save_pretrained(CFG.output_dir)