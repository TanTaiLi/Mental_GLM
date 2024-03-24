import torch
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModel

import sys
sys.path.append('/root/tuning_space/Components/')
import interact
from Static import prompt_dict, st, si

#查看每一层的“设备”，“类型”，“是否可以梯度下降”
#总参数量，可梯度下降参数量，可梯度下降参数量的百分比
def model_profile(model):
    # 初始化计数器
    total_params = 0
    trainable_params = 0

    # 查看每一层的参数
    for i, para in enumerate(model.named_parameters()):
        param_size = torch.prod(torch.tensor(para[1].size()))
        total_params += param_size

        # 检查参数是否可训练
        if para[1].requires_grad:
            trainable_params += param_size

        #print(f'{i}\t|\t{para[0]}\t|\t{para[1].device}\t|\t{para[1].dtype}\t|\t{para[1].requires_grad}')

    # 计算可训练参数的百分比
    trainable_percentage = (trainable_params / total_params) * 100

    print(f'Total Parameters: {total_params}')
    print(f'Trainable Parameters: {trainable_params}')
    print(f'Percentage of Trainable Parameters: {trainable_percentage:.2f}%')
    

#根据给定的model和tokenizer进行chat的函数
def chat(model, tokenizer, sys_prompt=False):
    input_ids = torch.cat([st['[gMASK]'], st['sop']],dim=1).cuda()
    if sys_prompt:
        input_ids = torch.cat([input_ids, st['<|system|>'], tokenizer.encode(prompt_dict['system_information'], return_tensors='pt').cuda()[:,2:], st['\n'], st['<|user|>']], dim=1)
    else:
        input_ids = torch.cat([input_ids, st['\n'], st['<|user|>']], dim=1)
    while True:
        input_ids = torch.cat([input_ids, tokenizer.encode(input('User:'), return_tensors='pt').cuda()[:,2:], st['\n'], st['<|assistant|>']], dim=1)
        input_ids = interact.inference_streamed(input_ids, model, tokenizer, in_nature_language=False, greedy=True)
        
        
def merge_lora(base_model_path, lora_path):
    # 载入基座模型
    base_model = AutoModel.from_pretrained(base_model_path, trust_remote_code=True).cuda().half()
    # 暂存用以验证权重是否改变
    first_weight = base_model.transformer.encoder.layers[0].self_attention.query_key_value.weight
    first_weight_old = first_weight.clone()
    
    # 载入lora结构的模型
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    
    # 合并lora结构
    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)
    
    # 验证结构
    assert not torch.allclose(first_weight_old, first_weight), 'Weight Should Change after Lora Merge'
    
    # 给模型改名
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model.state_dict().items()
        if "lora" not in k
    }
    
    return lora_model


def merge_lora_train(base_model, lora_path):
    # 载入lora结构的模型
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    # 合并lora结构
    lora_model = lora_model.merge_and_unload()
    lora_model.train(False)
    print('*'*50)
    print(f'成功merge:{lora_path}')
    print('*'*50)
    return lora_model