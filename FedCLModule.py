import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
)
import shutil
import os
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


class FedCLModule:
    """
    FedCLModule 用于在 PILoRA-cifar 框架中新增联邦持续学习功能，
    包括客户端侧的局部统计信息计算、选择性梯度更新和服务器端的聚合与元参数更新。

    参数说明：
    - model: 当前的 LoRA 模型（例如由 VITLORA 实例化的模型）
    - encoder: 预训练 encoder，用于提取输入样本的全局语义特征 E(x)
    - readout_fn: 用于提取样本的 readout 表示 F(x)，通常使用模型内部某层输出
    - lr: 学习率
    - tau: 动态阈值，控制知识迁移的触发条件（初始值）
    - gamma: 包含三个维度权重的元参数 (γ₁, γ₂, γ₃)，用于梯度对齐、全局语义和 readout 指标的加权
    - lambda_reg: 正则项权重，用于衡量选择性梯度更新误差
    - topk_ratio: 参数更新（梯度增量）压缩时保留绝对值最大的比例（如 0.1 表示保留 10%）
    """

    def __init__(self, model, encoder, readout_fn, args, accelerator, lr=0.01,
                 tau=0.5, lambda_reg=0.1, topk_ratio=0.1,
                 prev_meta_params=None, prev_similarity=None, prev_pold=None, prev_rold=None):

        self.g_model = model
        self.encoder = encoder
        self.readout_fn = readout_fn
        self.args = args
        self.accelerator = accelerator
        self.lr = 0.01 # TODO 弄成option可控制的
        self.alpha = 1
        self.historical_grad = None
        self.meta_iterations = 30 #TODO
        self.tau_history = []
        # 根据传入的 prev_meta_params 加载之前任务的元参数，否则使用默认值
        def to_val(x):
            # 如果 x 是 Tensor，则返回它的数值，否则直接返回 x（假定为 float)
            return x.item() if isinstance(x, torch.Tensor) else x

        if prev_meta_params is not None:
            self.meta_params = {
                'tau': torch.tensor(to_val(prev_meta_params['tau']), requires_grad=True,
                                    device=self.accelerator.device),
                # 'gamma1': torch.tensor(to_val(prev_meta_params['gamma1']), requires_grad=True,
                #                        device=self.accelerator.device),
                # 'gamma2': torch.tensor(to_val(prev_meta_params['gamma2']), requires_grad=True,
                #                        device=self.accelerator.device),
                # 'gamma3': torch.tensor(to_val(prev_meta_params['gamma3']), requires_grad=True,
                #                        device=self.accelerator.device),
            }
            self.tau = self.meta_params['tau']
            # self.gamma1 = self.meta_params['gamma1']
            # self.gamma2 = self.meta_params['gamma2']
            # self.gamma3 = self.meta_params['gamma3']
        else:
            self.tau = tau
            # self.gamma1, self.gamma2, self.gamma3 = gamma
            self.meta_params = {
                'tau': torch.tensor(self.tau, requires_grad=True, device=self.accelerator.device),
                # 'gamma1': torch.tensor(self.gamma1, requires_grad=True, device=self.accelerator.device),
                # 'gamma2': torch.tensor(self.gamma2, requires_grad=True, device=self.accelerator.device),
                # 'gamma3': torch.tensor(self.gamma3, requires_grad=True, device=self.accelerator.device)
            }

        if prev_similarity is not None and prev_similarity[0] is not None and prev_similarity[1] is not None:
            self.S_global = prev_similarity[0].to(self.accelerator.device)
            self.S_readout = prev_similarity[1].to(self.accelerator.device)
        else:
            self.S_global = torch.tensor(0.8, device=self.accelerator.device)
            self.S_readout = torch.tensor(0.7, device=self.accelerator.device)

        self.lambda_reg = lambda_reg
        self.topk_ratio = topk_ratio

        self.P_old = prev_pold
        self.R_old = prev_rold

        self.meta_optimizer = AdamW(list(self.meta_params.values()), lr=self.lr)
        self.meta_optimizer = self.accelerator.prepare(self.meta_optimizer)

    def _get_state_dict(self, model):
        """
        返回模型当前参数的 deep copy
        """
        state = {}
        for name, param in model.named_parameters():
            state[name] = param.detach().clone()
        return state

    def update_last_task(self, idx, current_task):
        import os
        client_dir = os.path.join(
            self.args.base_dir,
            f"seq_{self.args.idrandom}_seed{self.args.seed}",
            str(self.args.baseline),
            str(self.args.dataset),
            f"topK_{str(self.args.topk_ratio)}",
            f"client_idx_{idx}"
        )
        os.makedirs(client_dir, exist_ok=True)
        last_task_path = os.path.join(client_dir, 'last_task.txt')
        with open(last_task_path, 'w') as f:
            f.write(str(current_task))

    def compute_local_statistics(self, data_loader, model):
        """
        计算客户端本地统计信息，包括：
         - 各类别原型 P_i^(c) （公式 (1)）
         - 整体 readout 表示 R_i （公式 (2)）

        这里假设 data_loader 每个 batch 返回字典，包含至少键 "input_ids", "attention_mask" 和 "labels"
        """

        # 定义 encoder 和 readout_fn，根据模型结构进行适配
        # 适用于lora
        # full：model.module.model.model.model.encoder
        encoder = lambda input_ids, attention_mask: model.module.model.model.base_model.model.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        ).last_hidden_state[:, 0, :]

        readout_fn = lambda input_ids, attention_mask: model.module.model.model.base_model.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        ).decoder_hidden_states[-2].mean(dim=1)

        # 存储每个类别的特征列表
        prototypes = {}  # 格式: { label: [feature_vectors] }
        readout_list = []  # 存放所有样本的 readout 表示
        total_samples = 0

        model.eval()
        device = next(model.parameters()).device  # 获取模型所在的设备

        with torch.no_grad():
            for step, batch in enumerate(data_loader):
                # batch 是一个 dict，包含 "input_ids", "attention_mask", "labels"
                # 如果想让 input_ids / attention_mask 在 GPU 上进行计算，可以传到 device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                # 如果你需要在 CPU 上后处理 labels，也可以不转到 device
                labels = batch["labels"]  # 保持在 CPU

                # 1. 使用 encoder 提取全局语义特征 h (公式 (1))
                h = encoder(input_ids=input_ids, attention_mask=attention_mask)
                # 此处 h 的维度假设为 (batch, d_h)

                # 2. 使用 readout_fn 提取倒数第二层表示 r (公式 (2))
                r = readout_fn(input_ids=input_ids, attention_mask=attention_mask)
                # 此处 r 的维度假设为 (batch, d_r)

                batch_size = input_ids.size(0)
                total_samples += batch_size
                readout_list.append(r)

                # 将每个样本的特征按照标签分类存储
                for i, label in enumerate(labels):
                    lbl = int(label.item())
                    if lbl not in prototypes:
                        prototypes[lbl] = []
                    prototypes[lbl].append(h[i])

        # 对每个类别的特征取均值：  P_i^(c) = (1/N) Σ_j h_{ij}
        for lbl in prototypes:
            # 每个类别存了多个 feature vector，将其堆叠后再取 mean
            features = torch.stack(prototypes[lbl], dim=0)  # (N, d_h)
            prototypes[lbl] = features.mean(dim=0)  # (d_h)

        # 计算整体 readout 表示： R_i = (1/N) Σ r_{ij}
        readout_concat = torch.cat(readout_list, dim=0)  # (total_samples, d_r)
        R_i = readout_concat.mean(dim=0)

        return prototypes, R_i, total_samples

    def selective_gradient_update(self, loss, optimizer, model):
        """
        对一次 mini-batch 进行选择性梯度更新（公式 (3)-(7)），但不更新历史梯度
        返回： 本次 mini-batch 的原始梯度 g_curr, 以及本次更新后的总损失
        """
        optimizer.zero_grad()
        self.accelerator.backward(loss)


        trainable_params = [p for p in model.parameters() if p.requires_grad]

        device = trainable_params[0].device

        g_curr_list = [p.grad.detach().flatten() for p in trainable_params if p.grad is not None]
        g_curr = torch.cat(g_curr_list).to(device)



        # # 通过 autograd.grad 获取所有 trainable 参数的梯度向量
        # grads = torch.autograd.grad(loss, trainable_params, create_graph=True)
        # g_curr = torch.cat([g.view(-1) for g in grads if g is not None], dim=0).detach()

        if self.historical_grad is None:
            current_hist = g_curr.detach().clone()
        else:
            current_hist = self.historical_grad

        # 计算梯度对齐指标 φ_grad（公式 (3)）——可能存在数值不稳定问题
        # dot_product = torch.dot(g_curr, current_hist)
        # norm_product = torch.norm(g_curr) * torch.norm(current_hist) + 1e-8
        # phi_grad_1 = dot_product / norm_product

        # 解决方法：采用归一化
        g_curr_unit = g_curr / (torch.norm(g_curr) + 1e-8)
        current_hist_unit = current_hist / (torch.norm(current_hist) + 1e-8)
        current_hist_unit = current_hist_unit.to(g_curr_unit.device)
        phi_grad = torch.dot(g_curr_unit, current_hist_unit)

        # 计算综合相似性指标 φ_total = γ₁*φ_grad + γ₂*S_global + γ₃*S_readout （公式 (4)）
        # phi_total = self.gamma1 * phi_grad + self.gamma2 * self.S_global + self.gamma3 * self.S_readout

        # new
        phi_total = phi_grad + self.alpha * ((1-self.S_global) * self.S_readout)

        projected_component = (torch.dot(g_curr_unit, current_hist_unit) / (
                torch.norm(current_hist_unit) ** 2 + 1e-8)) * current_hist_unit

        s = torch.tensor(max(phi_total - self.tau, 0), dtype=torch.float32)  # 确保s为Tensor类型
        w = torch.clamp(s, 0, 1)  # 这里的裁剪操作确保 w 在 [0, 1] 之间

        g_proj = projected_component
        g_res = g_curr - g_proj

        g_transfer = w * g_proj + (1 - w) * g_res

        ptr = 0
        for p in trainable_params:
            if p.grad is None:
                continue  # 跳过无梯度的参数（如未参与计算的BN层）
            grad_size = p.grad.numel()
            p.grad.copy_(g_transfer[ptr:ptr + grad_size].view(p.grad.shape))  # 直接覆盖原始梯度
            ptr += grad_size

        # # 4. 必须调用Accelerator的backward（重点！）
        # self.accelerator.backward(loss, retain_graph=True)  # 传入原始loss，触发梯度同步和混合精度处理

        # # 根据 φ_total 与 τ 判断采取哪种梯度操作：
        # if phi_total > self.tau:
        #     # 梯度投影：  g_proj = (<g_curr, current_hist> / ||current_hist||²) * current_hist  （公式 (5)）
        #     projected_component = (torch.dot(g_curr_unit, current_hist_unit) / (
        #                 torch.norm(current_hist_unit) ** 2 + 1e-8)) * current_hist_unit
        #     w = torch.sigmoid(phi_total)  # 权重 w = σ(φ_total) （公式 (5)）
        #     g_transfer = w * projected_component
        # else:
        #     # 正交化：  g_transfer = g_curr - g_proj  （公式 (6)）
        #     projected_component = (torch.dot(g_curr_unit, current_hist_unit) / (
        #                 torch.norm(current_hist_unit) ** 2 + 1e-8)) * current_hist_unit
        #     g_transfer = g_curr - projected_component
        #
        # # 正则化损失：  reg_loss = λ * ||g_curr - g_transfer||² （公式 (7)）
        # reg_loss = self.lambda_reg * torch.norm(g_curr - g_transfer) ** 2

        # 将正则项加入原始 loss 中，并进行反向传播与更新
        # total_loss = loss + reg_loss
        # self.accelerator.backward(total_loss)
        optimizer.step()

        return loss.item(), g_curr, phi_grad

    def local_training(self, model, train_loader, optimizer, lr_scheduler, idx, current_output_dir, historical_grad=None,
                       local_ep=1,
                       current_task=0):
        """
        客户端本地训练接口：
          1. 计算局部统计信息（公式 (1) 和 (2)）。
          2. 构造优化器，仅优化 LoRA 层参数（若 args.is_peft 为 True），并使用 AdamW 及 lr_scheduler，
             参考 update_weights_local 的实现。
          3. 对 train_loader 中所有 mini-batch 进行训练，每个 mini-batch 调用 selective_gradient_update，
             并保存每个 mini-batch 的原始梯度。
          4. 所有 local epoch 完成后，统一更新历史梯度为所有 mini-batch 原始梯度的平均值。
          5. 计算模型参数更新增量 delta = current_state - initial_state，经 TopK 压缩后返回。
        """
        self.historical_grad = historical_grad

        if self.accelerator.is_main_process:
            logger.info("***** Running training in Local Client *****")
            logger.info(
                f"Client idx = {idx},  training size = {train_loader.total_dataset_length}")
            logger.info(
                f" Learning Rate = {self.args.encoders_lr}, Classifier Learning Rate = {self.args.classifier_lr},"
                f" Warmup Num = {self.args.num_warmup_steps}, Pre-trained Model = {self.args.model_name_or_path}")
            logger.info(
                f" Batch Size = {self.args.local_bs}, Local Epoch = {self.args.local_ep}")

        initial_state = self._get_state_dict(model)

        global_step = 0

        if self.accelerator.is_main_process:
            # Delete previous models if we do not want to save all checkpoints.
            if 'save_all_ckpt' not in self.args.baseline:
                for saved_output_dir in self.args.saved_output_dir[:-2]:  # We need -2 so that we can load model.
                    if os.path.isdir(saved_output_dir):
                        shutil.rmtree(saved_output_dir)
        if self.accelerator.is_main_process:
            print(100 * '#')
            print("Begin Local Training!")

        prototypes, R_i, total_samples = self.compute_local_statistics(train_loader, model)

        gradients = []
        phi_list = []
        for iter in range(local_ep):
            progress_bar = tqdm(range(len(train_loader)), disable=not self.accelerator.is_local_main_process)
            for batch_idx, inputs in enumerate(train_loader):
                outputs = model(**inputs, restrict_label=True)
                loss = outputs.loss
                loss_val, g_curr, phi_grad = self.selective_gradient_update(loss, optimizer, model)
                gradients.append(g_curr)
                phi_list.append(phi_grad)
                global_step += 1
                if lr_scheduler is not None:
                    lr_scheduler.step()
                progress_bar.update(1)
                progress_bar.set_description(
                    'Train Iter (Epoch=%3d,loss=%5.3f)' % (iter, loss_val))
        self.accelerator.wait_for_everyone()

        phi_avg = torch.mean(torch.stack(phi_list))
        self.historical_grad = torch.mean(torch.stack(gradients), dim=0).detach()

        # 保存当前任务的历史梯度到文件
        save_dict = {"historical_avg_grad": self.historical_grad}
        output_file_path = os.path.join(current_output_dir, 'historical_avg_grad.pt')
        self.accelerator.save(save_dict, output_file_path)
        logger.info(f"Local historical_grad saved to {output_file_path}")

        # 调用已定义的 update_last_task 方法更新 last_task 文件（记录当前任务编号）
        self.update_last_task(idx, current_task)

        # 7. 计算模型参数更新增量 delta = current_state - initial_state
        current_state = self._get_state_dict(model)
        delta_model = {}
        for key in current_state:
            delta_model[key] = current_state[key] - initial_state[key]

        #8. 压缩
        if self.accelerator.is_main_process:
            print('Doing compress by TopK, ratio is {}'.format(self.topk_ratio))
        delta_model_compressed = self.global_topk_compress_lora(delta_model, self.topk_ratio, model)
        # 这里是否需要更新？
        # self.initial_state = current_state
        model = self.accelerator.unwrap_model(model)
        model.cpu()
        return prototypes, R_i, total_samples, delta_model_compressed, phi_avg

    def topk_compress(self, delta_model, k_ratio):
        """
        对 delta_model 进行 TopK 压缩，仅保留每个 tensor 中 k_ratio 比例绝对值最大的元素，其余置零。
        """
        compressed_delta = {}
        for name, tensor in delta_model.items():
            tensor_flat = tensor.view(-1)
            total_elements = tensor_flat.numel()
            k_num = max(int(total_elements * k_ratio), 1)
            k_num = min(k_num, total_elements)  # 防止 k_num 大于张量尺寸
            # 直接通过 topk 得到要保留的索引
            topk_values, topk_indices = torch.topk(torch.abs(tensor_flat), k_num, sorted=False)
            # 创建一个全零张量，并将 topk 部分还原
            compressed = torch.zeros_like(tensor_flat)
            compressed.scatter_(0, topk_indices, tensor_flat[topk_indices])
            compressed_delta[name] = compressed.view_as(tensor)
        return compressed_delta

    def global_topk_compress(self, delta_model, k_ratio):
        """ 对模型参数更新 delta_model 进行全局 TopK 压缩， 仅保留所有层中绝对值最大的 k_ratio 比例的元素，其余置零。
        参数：
          - delta_model: dict 格式，键为参数名，值为对应的更新 tensor。
          - k_ratio: float，例如 0.1 表示保留全局更新中绝对值最大的 10% 元素。
        返回：
          - compressed_delta: 与 delta_model 格式相同，但仅保留全局 TopK 更新元素，其余置零。
        """
        # 将所有层的更新扁平化后拼接成一个大向量
        all_updates = []
        for name, tensor in delta_model.items():
            all_updates.append(tensor.view(-1))
        all_updates = torch.cat(all_updates, dim=0)

        total_elements = all_updates.numel()
        k_num = max(int(total_elements * k_ratio), 1)
        k_num = min(k_num, total_elements)

        # 利用 global TopK 找到绝对值最大的 k 个元素中的最小值，即全局阈值
        topk_values = torch.topk(torch.abs(all_updates), k_num, sorted=False)[0]
        global_threshold = topk_values.min()

        # 对每个层做筛选，保留绝对值大于等于 global_threshold 的更新值
        compressed_delta = {}
        for name, tensor in delta_model.items():
            mask = torch.ge(torch.abs(tensor), global_threshold)
            compressed_delta[name] = tensor * mask.float()

        return compressed_delta

    def global_topk_compress_lora(self, delta_model, k_ratio, model):
        """ 对模型参数更新 delta_model 进行全局 TopK 压缩， 仅对 LoRA 层进行压缩，其他层保持不变。
        参数：
          - delta_model: dict 格式，键为参数名，值为对应的更新 tensor。
          - k_ratio: float，例如 0.1 表示保留全局更新中绝对值最大的 10% 元素。
        返回：
          - compressed_delta: 与 delta_model 格式相同，但仅对 LoRA 层进行 TopK 压缩，其他层保持不变。
        """
        # 收集所有带有 'lora' 字符串的层名称
        lora_layer_names = []
        if self.args.is_peft:
            for name, param in model.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    lora_layer_names.append(name)
        else:
            # 如果不是 PEFT 模式，所有参数都参与压缩
            for name, param in model.named_parameters():
                if param.requires_grad:
                    lora_layer_names.append(name)

        # 将所有 LoRA 层的更新扁平化后拼接成一个大向量
        all_updates = []
        for name, tensor in delta_model.items():
            if name in lora_layer_names:  # 只对 LoRA 层进行压缩
                all_updates.append(tensor.view(-1))

        if all_updates:
            all_updates = torch.cat(all_updates, dim=0)

            total_elements = all_updates.numel()
            k_num = max(int(total_elements * k_ratio), 1)
            k_num = min(k_num, total_elements)

            # 利用 global TopK 找到绝对值最大的 k 个元素中的最小值，即全局阈值
            topk_values = torch.topk(torch.abs(all_updates), k_num, sorted=False)[0]
            global_threshold = topk_values.min()

            # 对每个层做筛选，保留绝对值大于等于 global_threshold 的更新值
            compressed_delta = {}
            for name, tensor in delta_model.items():
                if name in lora_layer_names:
                    mask = torch.ge(torch.abs(tensor), global_threshold)
                    compressed_delta[name] = tensor * mask.float()
                else:
                    # 对非 LoRA 层不进行压缩，直接保留原值
                    compressed_delta[name] = tensor
        else:
            # 如果没有 LoRA 层，就返回原模型
            compressed_delta = delta_model

        return compressed_delta

    def server_aggregate_and_meta_update(self, client_updates, client_prototypes, client_readouts, client_sample_counts,
                                         client_phi, lambda_KD=0.1, lambda_conf=0.1, update_history=False):

        if self.accelerator.is_main_process:
            logger.info("***** Begin to aggregate and update in Server *****")


        if self.P_old is not None and self.R_old is not None:
            P_old = self.P_old
            R_old = self.R_old
        else:
            P_old = None
            R_old = None

        # 公共步骤：聚合客户端原型（按样本数加权平均）
        global_prototypes = {}
        total_counts = {}
        for cid, proto_dict in client_prototypes.items():
            count = client_sample_counts[cid]
            for label, feat in proto_dict.items():
                if label not in global_prototypes:
                    global_prototypes[label] = feat * count
                    total_counts[label] = count
                else:
                    global_prototypes[label] += feat * count
                    total_counts[label] += count
        for label in global_prototypes:
            global_prototypes[label] /= total_counts[label]

        total_samples = sum(client_sample_counts.values())
        # 聚合客户端 readout 表示（按样本数加权平均）
        R_new = sum(client_readouts[cid] * client_sample_counts[cid] for cid in client_readouts) / total_samples
        # 计算全局任务整体原型 P_new（所有类别原型均值）
        P_new = sum(global_prototypes[label] for label in global_prototypes) / len(global_prototypes)
        k = []
        # 针对是否存在上一个任务进行处理
        if P_old is not None and R_old is not None:
            P_old = P_old.to(P_new.device)
            R_old = R_old.to(R_new.device)

            phi_target = torch.tensor(0.5, device=P_new.device)  # 关键：改为可学习参数或配置项
            S_global = F.cosine_similarity(P_new.unsqueeze(0), P_old.unsqueeze(0))
            S_readout = F.cosine_similarity(R_new.unsqueeze(0), R_old.unsqueeze(0))
            kd_loss = torch.norm(P_new - P_old) ** 2

            for _ in range(self.meta_iterations):

                # 计算目标对齐项 L_target（指数线性单元）
                elu_sum = 0.0
                for cid, phi_val in client_phi.items():
                    weight = client_sample_counts[cid] / total_samples
                    elu_sum += weight * F.elu(phi_val - self.meta_params['tau'])
                avg_elu = elu_sum / len(client_phi) if client_phi else 0.0  # 避免除零
                L_target = torch.square(torch.relu(avg_elu - phi_target))  # [·]_+^2

                # 计算冲突抑制项 L_conflict（Hinge Loss平方项）
                conflict_sum = 0.0
                for cid, phi_val in client_phi.items():
                    weight = client_sample_counts[cid] / total_samples
                    conflict_sum += weight * torch.square(torch.relu(self.meta_params['tau'] - phi_val))
                L_conflict = lambda_conf * (conflict_sum / max(len(client_phi), 1))  # 防零除

                meta_loss = kd_loss + L_target + L_conflict

                # # 计算全局代理迁移触发比例: phi_global_proxy = 1/N * sum_i sigmoid( gamma1*phi_i + gamma2*S_global_new + gamma3*S_readout_new - tau )
                # phi_global_proxy = 0
                # for cid, phi_val in client_phi.items():
                #     weight = client_sample_counts[cid] / total_samples
                #     temp = self.meta_params['gamma1'] * phi_val.detach() + self.meta_params['gamma2'] * S_global + \
                #            self.meta_params['gamma3'] * S_readout - self.meta_params['tau']
                #     phi_global_proxy += weight * torch.sigmoid(temp)
                # # 设定目标迁移触发比例 phi_target（例如 0.5）
                # phi_target = torch.tensor(0.5, device=phi_global_proxy.device)
                # # 迁移冲突项: L_conflict = (phi_global_proxy - phi_target)^2
                # L_conflict = (phi_global_proxy - phi_target) ** 2
                #
                # L_reg = self.lambda_reg * (
                #             self.meta_params['gamma1'] ** 2 + self.meta_params['gamma2'] ** 2 + self.meta_params[
                #         'gamma3'] ** 2)
                #
                # meta_loss = lambda_KD * kd_loss + lambda_conf * L_conflict + L_reg

                self.meta_optimizer.zero_grad()
                self.accelerator.backward(meta_loss)

                self.meta_optimizer.step()
                k.append(self.meta_params['tau'])

            # 提取最终tau（建议保留历史记录）
            self.tau = self.meta_params['tau'].item()
            # 新增：记录优化轨迹（可选）
            self.tau_history.append(self.tau)
        else:
            # 对于第一个任务，未进行元参数更新，直接使用当前计算得到的 P_new 和 R_new
            S_global, S_readout = None, None

        P_new = P_new.detach()
        R_new = R_new.detach()

        # 通用步骤：聚合客户端参数更新
        aggregated_delta = {}
        for cid, delta in client_updates.items():
            weight = client_sample_counts[cid] / total_samples
            for name, value in delta.items():
                aggregated_delta[name] = aggregated_delta.get(name, 0) + weight * value
        # 删除module前缀
        corrected_delta = {}
        for key, value in aggregated_delta.items():
            new_key = key[len("module."):] if key.startswith("module.") else key
            corrected_delta[new_key] = value
        for name, param in self.g_model.named_parameters():
            if name in corrected_delta:
                param.data += corrected_delta[name].to(param.device)


        # 调用保存接口，将当前全局原型、readout以及（若有）元参数保存下来
        # 假设 save_old_param 已经封装保存逻辑，这里传入 (P_old_new, R_old_new, S_global, S_readout)
        # self.save_old_param(P_old_new, R_old_new, S_global, S_readout)

        return self.tau, P_new, R_new, self.g_model, S_global, S_readout
