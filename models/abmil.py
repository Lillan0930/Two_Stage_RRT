"""
ABMIL (Attention-based Multiple Instance Learning)

包含标准ABMIL和门控ABMIL，从 mm_rrt_abmil.py 拆分出来。
统一返回格式为 dict：{'logits', 'Y_prob', 'Y_hat', 'attention'}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mil_registry import register_mil
# initialize_weights defined locally


class ABMILBase(nn.Module):
    """ABMIL基类 — 处理batched/sequential输入的通用逻辑"""

    def forward(self, x):
        """统一处理不同形状的输入"""
        if isinstance(x, list):
            results = [self._forward_single(bag) for bag in x]
            return self._stack_results(results)
        elif len(x.shape) == 3:
            batch_size = x.shape[0]
            results = [self._forward_single(x[i]) for i in range(batch_size)]
            return self._stack_results(results)
        else:
            return self._forward_single(x)

    def _forward_single(self, x):
        raise NotImplementedError

    @staticmethod
    def _stack_results(results):
        """将多个单样本结果堆叠为batch"""
        logits = torch.stack([r['logits'].squeeze(0) for r in results])
        Y_prob = torch.stack([r['Y_prob'].squeeze(0) for r in results])
        Y_hat = torch.stack([r['Y_hat'] for r in results])
        # attention 长度可能不一致，保持list
        attention = [r['attention'] for r in results]
        return {
            'logits': logits,
            'Y_prob': Y_prob,
            'Y_hat': Y_hat,
            'attention': attention,
        }


@register_mil("abmil")
class AttentionMIL(ABMILBase):
    """标准ABMIL — Tanh注意力"""

    def __init__(self, input_dim=512, hidden_dim=128, num_classes=4, dropout_rate=0.25):
        super(AttentionMIL, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.apply(initialize_weights)

    def _forward_single(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(0)
        if len(x.shape) != 2:
            raise ValueError(f"Expected 2D input [N, C], got shape {x.shape}")

        A = self.attention(x)          # [N, 1]
        A = A.transpose(0, 1)          # [1, N]
        A = F.softmax(A, dim=1)        # [1, N]

        Z = torch.matmul(A, x)         # [1, C]
        logits = self.classifier(Z)    # [1, num_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.argmax(Y_prob, dim=1)

        return {
            'logits': logits,
            'Y_prob': Y_prob,
            'Y_hat': Y_hat,
            'attention': A.squeeze(0),
        }


@register_mil("gated_abmil")
class GatedAttentionMIL(ABMILBase):
    """门控ABMIL (Ilse et al. 2018) — Tanh + Sigmoid门控"""

    def __init__(self, input_dim=512, hidden_dim=128, num_classes=4, dropout_rate=0.25):
        super(GatedAttentionMIL, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.attention_V = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_w = nn.Linear(hidden_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.apply(initialize_weights)

    def _forward_single(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(0)

        A_V = self.attention_V(x)      # [N, hidden_dim]
        A_U = self.attention_U(x)      # [N, hidden_dim]
        A = self.attention_w(A_V * A_U)  # [N, 1]
        A = A.transpose(0, 1)          # [1, N]
        A = F.softmax(A, dim=1)        # [1, N]

        Z = torch.matmul(A, x)         # [1, C]
        logits = self.classifier(Z)    # [1, num_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.argmax(Y_prob, dim=1)

        return {
            'logits': logits,
            'Y_prob': Y_prob,
            'Y_hat': Y_hat,
            'attention': A.squeeze(0),
        }

# Local initialize_weights (used by ABMILBase)
def initialize_weights(module):
    # Single-module init — always called via .apply(), which recurses; see
    # models/mm_rrt_encoder.initialize_weights for the full rationale.
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None: module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None: module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
