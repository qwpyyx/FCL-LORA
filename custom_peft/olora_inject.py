import torch.nn as nn
from peft.tuners.lora import LoraConfig
from custom_peft.olora_layers import OLoRALinear

def get_olora_model(model: nn.Module, lora_config: LoraConfig, args, adapter_name: str = "default"):
    target_modules = lora_config.target_modules
    fan_in_fan_out = lora_config.fan_in_fan_out
    r = lora_config.r
    lora_alpha = lora_config.lora_alpha
    dropout = lora_config.lora_dropout

    for name, module in model.named_modules():
        if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
            parent_module = get_parent(model, name)
            child_name = name.split('.')[-1]
            original = getattr(parent_module, child_name)

            new_module = OLoRALinear(
                in_features=original.in_features,
                out_features=original.out_features,
                r=r,
                lora_alpha=lora_alpha,
                dropout=dropout,
                fan_in_fan_out=fan_in_fan_out,
                r_sum=args.task * r
            )

            # 拷贝权重（只要主干权重，LoRA 是增量）
            new_module.weight = original.weight
            if hasattr(original, "bias"):
                new_module.bias = original.bias

            setattr(parent_module, child_name, new_module)

    return model

def get_parent(model: nn.Module, module_name: str):
    names = module_name.split(".")
    parent = model
    for name in names[:-1]:
        parent = getattr(parent, name)
    return parent
