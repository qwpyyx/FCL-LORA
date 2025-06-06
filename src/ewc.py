import torch
import torch.nn as nn


class EWC:
    def __init__(self, model, lamda=1000):
        self.model = model
        self.lamda = lamda
        self.prev_params = None
        self.fisher_information = None

    def compute_fisher_information(self, dataloader, loss_fn=nn.CrossEntropyLoss(), is_peft=False):
        """
        计算Fisher信息矩阵
        """
        fisher_information = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}

        # 判断是否使用LoRA微调
        if is_peft:
            # 只计算LoRA相关的参数的Fisher信息
            for name, param in self.model.named_parameters():
                if 'lora' in name.lower():  # 假设LoRA相关的参数包含'lora'关键字
                    fisher_information[name] = torch.zeros_like(param)
        else:
            # 计算全量微调的Fisher信息
            for name, param in self.model.named_parameters():
                fisher_information[name] = torch.zeros_like(param)

        # 计算模型的梯度
        self.model.eval()
        for inputs, labels in dataloader:
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()

            for name, param in self.model.named_parameters():
                fisher_information[name] += param.grad ** 2 / len(dataloader)

        self.fisher_information = fisher_information

    def save_params(self, is_peft=False, save_path="model_params.pth"):
        """
        保存模型参数
        """
        model_state_dict = {}

        # 判断是否使用LoRA微调
        if is_peft:
            # 只保存LoRA相关的参数
            for name, param in self.model.named_parameters():
                if 'lora' in name:  # 假设LoRA相关的参数包含'lora'关键字
                    model_state_dict[name] = param.data
        else:
            # 保存全量微调的所有参数
            for name, param in self.model.named_parameters():
                model_state_dict[name] = param.data

        # 保存参数到文件
        torch.save(model_state_dict, save_path)
        print(f"Parameters saved to {save_path}")

    def ewc_loss(self, is_peft=False):
        """
        计算EWC损失项
        """
        loss = 0
        if self.prev_params and self.fisher_information:
            for name, param in self.model.named_parameters():
                # 判断是否为LoRA微调
                if is_peft and 'lora' not in name:  # 只计算LoRA相关参数
                    continue

                fisher = self.fisher_information[name]
                prev_param = self.prev_params[name]
                loss += (fisher * (param - prev_param) ** 2).sum()

        return self.lamda * loss