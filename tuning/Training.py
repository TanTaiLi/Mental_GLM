import pickle
import shutil
import torch
import deepspeed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
import os
import sys
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter 

sys.path.append('/root/tuning_space/Components/')
import model_tools
import utils
from Config import CFG, Deepspeed_config

if __name__ == "__main__":
    # 载入模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(CFG.model_path, trust_remote_code=True).half()

    if CFG.lora_sft:
        # 应用PEFT LoRA
        print('conducting peft lora ---------------')
        if CFG.preMerge_lora_list:
            for lora_path in CFG.preMerge_lora_list:
                model = model_tools.merge_lora_train(model, lora_path)
        if CFG.continue_training:
            print(f'继续训练：{CFG.continue_training}')
            model=PeftModel.from_pretrained(model, CFG.continue_training)
            for name, param in model.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
        else:
            model = get_peft_model(model, CFG.Lora_config)
    
    model_tools.model_profile(model)
    # 加载Deepspeed config
    with open(CFG.Deepspeed_config_path, 'w') as f:
        json.dump(Deepspeed_config, f, indent=4)
    
    # DeepSpeed初始化
    engine, optimizer, training_dataloader, lr_schedule = deepspeed.initialize(
        model=model,
        config=CFG.Deepspeed_config_path
    )

    # 创建分布式采样器和数据加载器
    finetuning_instruction = utils.conversation_dataset(
        CFG.data_list,
        tokenizer,
        CFG.max_tokens,
        CFG.query_key,
        CFG.answer_key,
        CFG.max_sample_list,
        CFG.num_workers
    )
    
    sampler = DistributedSampler(finetuning_instruction)
    instruction_loader = DataLoader(
        finetuning_instruction,
        shuffle=False,
        batch_size=CFG.batch_size,
        sampler=sampler,
        collate_fn=utils.coll_fn
    )
    print('dataloader的长度：', len(instruction_loader))

    engine.train()
    losses = []  # loss收集表
    log_interval = CFG.record_steps  # 记录loss的interval
    save_interval = CFG.save_steps
    
    if os.path.exists(CFG.tensorboard_log_dir):
        shutil.rmtree(CFG.tensorboard_log_dir)
    writer = SummaryWriter(log_dir=CFG.tensorboard_log_dir)  # 创建一个SummaryWriter实例，指定日志目录


    for epoch in range(CFG.num_train_epochs):
        print(f'Epoch {epoch}')
        sampler.set_epoch(epoch)

        for step, batch in tqdm(enumerate(instruction_loader), total=len(instruction_loader)):
            outputs = engine(**batch)
            loss = outputs.loss
            if CFG.max_train_steps:
                if step >= CFG.max_train_steps:
                    break
            if step % log_interval == 0:
                losses.append(loss.item())  # 记录一次损失
                print(f"Step {step} - Loss: {loss.item()}")
                writer.add_scalar('Loss/train', loss.item(), step)  # 将损失记录到TensorBoard
            if step % save_interval == 0 and step != 0: # 保存一次模型
                model.save_pretrained(f"{CFG.output_dir}_epoch{epoch}_step{step}")
            engine.backward(loss)
            engine.step()
    writer.close()

    # 训练结束后绘制损失曲线
    with open(CFG.graph_name+'.pkl', 'wb') as f:
        pickle.dump(losses, f)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(losses)*log_interval, log_interval), losses, label='Training Loss')
    plt.title('Loss Curve')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(CFG.graph_name+'.pdf', format='pdf')
    plt.show()


    # 模型保存
    model.save_pretrained(CFG.output_dir)
    if CFG.shutdown:
        os.system("/usr/bin/shutdown")