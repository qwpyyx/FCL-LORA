#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import numpy as np
from torch.utils.data import Subset, DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from CPN import *
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DataCollatorWithPadding
from transformers import Trainer
import torch
from torch.cuda.amp import autocast, GradScaler


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        # dataset 是一个包含多字段（如 'input_ids', 'attention_mask', 'label'）的 DatasetDict 对象
        self.dataset = dataset.select(idxs)
        # 直接存储所有标签以便后续使用
        self.labels = [self.dataset[i]['labels'] for i in range(len(self.dataset))]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 获取当前索引处的数据，并确保它是一个字典类型，包含所有字段
        example = self.dataset[index]

        return {
            'input_ids': example['input_ids'],
            'attention_mask': example['attention_mask'],
            'labels': example['labels']
        }

    def get_all_labels(self):
        return self.labels


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, tokenizer, data_collator):
        self.args = args

        # # 使用 tokenizer 对数据集进行编码，然后再将数据集传给 DatasetSplit
        # dataset = dataset.map(
        #     lambda example: tokenizer(
        #         example['input_text'],
        #         truncation=True,
        #         padding='max_length',
        #         max_length=128
        #     ),
        #     batched=True
        # )
        #
        # # 然后创建 DatasetSplit 子集
        # self.client_dataset = DatasetSplit(dataset, idxs)
        # # print(f"Length of DatasetSplit: {len(self.dataset)}")
        # self.tokenizer = tokenizer
        # self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.client_dataset = Subset(dataset, idxs)
        # 使用 DataLoader 加载数据集
        self.trainloader = DataLoader(self.client_dataset, batch_size=self.args.local_bs, shuffle=True, num_workers=0,
                                      collate_fn=data_collator)


    def update_weights(self, model, lr):

        model.train()
        scaler = GradScaler()
        network_params = []
        if self.args.is_peft:
            for name, param in model.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    network_params.append({'params': param, 'lr': lr, 'weight_decay': 0.00001})
        else:
            for param in model.parameters():
                network_params.append({'params': param, 'lr': lr, 'weight_decay': 0.00001})

        optimizer = torch.optim.Adam(network_params)
        loss_fct = torch.nn.CrossEntropyLoss()

        # Local epoch
        for iter in range(self.args.local_ep):
            lee =[]
            for batch_idx, batch in enumerate(self.trainloader):
                inputs = {
                'input_ids': batch['input_ids'].to(self.args.device),
                'attention_mask': batch['attention_mask'].to(self.args.device),
                'labels': batch['labels'].to(self.args.device)
                }
                # decoder_input_ids = labels
                model.zero_grad()

                with autocast():
                    logits = model(**inputs)

                    loss_dce = loss_fct(logits, inputs['labels'])
                    lee.append(loss_dce.item())

                scaler.scale(loss_dce).backward()
                scaler.step(optimizer)
                scaler.update()

        return model.state_dict(), None
