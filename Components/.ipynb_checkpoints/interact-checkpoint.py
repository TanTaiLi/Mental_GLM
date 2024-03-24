import sys
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModel


sys.path.append('/root/tuning_space/Components/')
import interact
from Static import si

class CFG:
    max_length = 1000

def inference_streamed(prompt, model, tokenizer, device='cuda', in_nature_language=True, greedy=False):
    '''
    use case
        sample_prompt='用户:你好,请问你叫什么？\n 助手:\n'
        inference_streamed(sample_prompt, model, tokenizer)
    '''
    print('\nAssistant: ')
    # 将自然语言形式的prompt进行tokenlize并转化为tensor, 或者输入即为token_id
    if in_nature_language:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    else:
        input_ids = prompt
    # 流式生成predict token
    max_length = CFG.max_length # 设置最大生成token长度
    with torch.no_grad(): # 禁用梯度下降
        for count in range(max_length):
            # 获得最后一个词位置上的词表概率分布
            outputs = model(input_ids=input_ids) # 直接使用了torch的inference方法，得到的是ModelOuput对象
            next_token_logits = outputs.logits[:, -1, :] # 获取GPT模型最后一层的数据，并提取最后一个位置的词表概率分布
            
            # 从概率分布中抽象选择被predict的token;此处并不随机选一个子集，而是在所有词汇中都进行选择，也就是top_p=1
            if greedy:
                next_token = torch.argmax(torch.nn.functional.softmax(next_token_logits, dim=-1), dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(torch.nn.functional.softmax(next_token_logits, dim=-1), num_samples=1)
            
            # 判断是否结束
            if next_token.item() == tokenizer.eos_token_id or next_token.item() in si.values():
                print('\n')
                break
            
            # 解码predict token
            next_token_nl = tokenizer.decode([next_token.squeeze().tolist()]).replace("<0x0A>", "\n")
            print(next_token_nl, end='', flush=True)
                
            # 整合下一个input
            input_ids = torch.cat([input_ids, next_token], dim=1)
            

                
    return input_ids