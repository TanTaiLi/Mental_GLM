from peft import LoraConfig

class CFG:
    model_path = '/root/autodl-tmp/weights/chatglm3-6b' # model和tokenizer的原地址
    output_dir = '/root/autodl-tmp/checkpoints/full_params/full_params_sft-attempt-1' # checkpoint的存储地址
    Deepspeed_config_path = 'ds_config.json'
    graph_name = 'regular_tuning_loss_curve'
    tensorboard_log_dir = '/root/tf-logs/full_params_sft-attempt-1' #Rank16-ctx512-DeDu-280K
    
    lora_sft = False
    ##设定要提前并入 original model 的 lora结构，没有直接不填。和continue_training不冲突，continue_training是决定最后一层lora的行为
    preMerge_lora_list = [] 
    ###'/root/autodl-tmp/checkpoints/Rank16-ctx512-regular2-12w-addApi-15k_epoch0_step8000' #继续训练的lora地址，None则为不继续
    continue_training = None #'/root/autodl-tmp/checkpoints/previous/Rank16-ctx512-regular2-12w-addApi-15k_epoch0_step8000'
    shutdown = False
    
    num_train_epochs = 1
    max_train_steps = None # 最大的 step数，如果为 None，则默认不限制
    record_steps = 1
    save_steps = 1000
    
    batch_size = 1 # 需要和 micro_batch_size同步
    max_tokens = 512
    max_query = 64
    
    data_list = [
        #'/root/autodl-tmp/dataset/instill.jsonl',
        #'/root/autodl-tmp/dataset/smile/smile_conversation.jsonl', 
        #"/root/autodl-tmp/dataset/smile/smile_conversation_750.jsonl",
        #'/root/autodl-tmp/dataset/OESD-GG-zh_cn-1/half_query.jsonl', 
        #'/root/autodl-tmp/dataset/OESD-GG-zh_cn-1/single_query.jsonl', 
        #'/root/autodl-tmp/dataset/OESD-GG-zh_cn-1/2500_query.jsonl',
        #'/root/autodl-tmp/dataset/zhihu_qa/zhihu_qa_200.jsonl',
        #'/root/autodl-tmp/dataset/OESD-GG-zh_cn-1/api_gen_conversation.jsonl',
        #'/root/autodl-tmp/dataset/alpaca_chinese_dataset/chinese_data.jsonl',
        #'/root/autodl-tmp/dataset/GPT3.5-3270k/3270k-qa.jsonl',
        '/root/autodl-tmp/dataset/3270k-qa-Deduplicate-289k.jsonl',
    ]
    query_key = ['User', 'client']
    answer_key = ['Assisstant', 'counselor', 'Assistant', 'Assistance']
    max_sample_list = [280000]
    num_workers = 24
    
    # LoRA配置
    Lora_config = LoraConfig (
        r=16,
        lora_alpha=32,
        target_modules=["query_key_value"],  
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
Deepspeed_config = {
    "train_batch_size": 24,
    "gradient_accumulation_steps": 3,
    "train_micro_batch_size_per_gpu": 4,
    "steps_per_print": 20,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-3
        }
    },
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-3,
            "warmup_num_steps": 2e3,
            "total_num_steps": 8e3,
        }
    },
    "gradient_clipping": 1.0,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "cpu_offload": True
    }
}

# Stage3 最省资源的一集
Deepspeed_config = {
    "train_batch_size": 6,
    "gradient_accumulation_steps": 3,
    "train_micro_batch_size_per_gpu": 1,
    "steps_per_print": 20,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.001
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 0.001,
            "warmup_num_steps": 1000
        }
    },
    "gradient_clipping": 1.0,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    },
}

# Stage2
Deepspeed_config = {
    "train_batch_size": 6,
    "gradient_accumulation_steps": 3,
    "train_micro_batch_size_per_gpu": 1,
    "steps_per_print": 20,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.001
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 0.001,
            "warmup_num_steps": 2000
        }
    },
    "gradient_clipping": 1.0,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "cpu_offload": True
    }
}

# Stage1
Deepspeed_config = {
    "train_batch_size": 24,
    "gradient_accumulation_steps": 3,
    "train_micro_batch_size_per_gpu": 4,
    "steps_per_print": 2000,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 0.001
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 0.001,
            "warmup_num_steps": 1000
        }
    },
    "gradient_clipping": 1.0,
    "fp16": {
        "enabled": True
    },
    "zero_optimization": {
            "stage": 1,
            "cpu_offload": True
        },
}