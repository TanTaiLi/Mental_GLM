prompt_dict={
    'system_information':"你是ChatGLM3，一个由清华大学实验室训练的大型语言模型。你的设计和训练使你能够理解和生成类似人类的文本，以此来与用户进行深度的对话。在这个对话中，你的角色是一位心理治疗师，你的目标是为用户提供心理疏导。你需要尽可能地理解用户的情绪和问题，然后提供有帮助的反馈和建议。你的回答应该基于心理学的理论和实践，同时也需要展现出同理心和理解。你的目标是帮助用户更好地理解他们自己的情绪，提供策略来应对生活中的压力，以及提供一种方式来帮助他们在困难时期找到希望。然而，你必须始终记住，虽然你会尽力提供专业的心理咨询建议，你并非真正的心理治疗专家。你的建议和反馈都是基于你的训练数据，而不是基于专业的医疗知识或个人经验。因此，如果用户表现出严重的心理健康问题，如抑郁症、焦虑症或自杀倾向，你必须强调寻求专业的医疗帮助的重要性。你不能替代专业的心理治疗或医疗干预，你的角色是辅助和补充，而不是替代。你的所有回复都需要使用中文，当然必要的时候你可以回复一些英文。"
}

import torch

# Special token
st = {
    "[MASK]": torch.tensor([[64789]]).to('cuda'),
    "[gMASK]": torch.tensor([[64790]]).to('cuda'),
    "[sMASK]": torch.tensor([[64791]]).to('cuda'),
    "sop": torch.tensor([[64792]]).to('cuda'),
    "eop": torch.tensor([[64793]]).to('cuda'),
    "<|system|>": torch.tensor([[64794]]).to('cuda'),
    "<|user|>": torch.tensor([[64795]]).to('cuda'),
    "<|assistant|>": torch.tensor([[64796]]).to('cuda'),
    "<|observation|>": torch.tensor([[64797]]).to('cuda'),
    "\n": torch.tensor([[13]]).to('cuda')
}



# Special token CPU
stc = {
    "[MASK]": torch.tensor([[64789]]),
    "[gMASK]": torch.tensor([[64790]]),
    "[sMASK]": torch.tensor([[64791]]),
    "sop": torch.tensor([[64792]]),
    "eop": torch.tensor([[64793]]),
    "<|system|>": torch.tensor([[64794]]),
    "<|user|>": torch.tensor([[64795]]),
    "<|assistant|>": torch.tensor([[64796]]),
    "<|observation|>": torch.tensor([[64797]]),
    "\n": torch.tensor([[13]])
}


# Stop index
si = {
    "[gMASK]": 64790,
    "sop": 64792,
    "eop": 64793,
    "<|system|>": 64794,
    "<|user|>": 64795,
    "<|assistant|>": 64796,
    "<|observation|>": 64797,
    "\n": 13,
}