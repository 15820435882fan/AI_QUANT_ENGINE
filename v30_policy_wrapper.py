# v30_policy_wrapper.py
#
# V30 Transformer policy wrapper.
# 用于加载已训练好的 V30 Transformer 模型，并对单个序列做动作预测。

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from v30_model import build_v30_transformer


@dataclass
class V30PolicyConfig:
    device: str = "auto"


class V30TransformerPolicy:
    """
    封装好的策略类：
      - 从 checkpoint 加载 Transformer 模型
      - 提供 predict_action() / predict_risk() 接口
    """

    def __init__(self, model_path: str, config: Optional[V30PolicyConfig] = None):
        if config is None:
            config = V30PolicyConfig()
        self.config = config

        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        ckpt = torch.load(model_path, map_location=self.device)
        self.seq_len = int(ckpt.get("seq_len", 128))
        self.feature_dim = int(ckpt.get("feature_dim", 9))

        # 这里假定动作=4类，风险=3类，与 v30_train_supervised 中一致
        self.model: nn.Module = build_v30_transformer(
            feature_dim=self.feature_dim,
            num_actions=4,
            num_risks=3,
            seq_len=self.seq_len,
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict_logits(self, seq_np) -> tuple:
        """
        seq_np: [seq_len, feature_dim] 的 numpy 数组
        返回:
          logits_action: torch.Tensor [num_actions]
          logits_risk:   torch.Tensor [num_risks]
        """
        x = torch.tensor(seq_np, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1, T, F]
        logits_action, logits_risk = self.model(x)
        return logits_action.squeeze(0), logits_risk.squeeze(0)

    @torch.no_grad()
    def predict_action(self, seq_np) -> int:
        logits_action, _ = self.predict_logits(seq_np)
        return int(logits_action.argmax(dim=-1).item())

    @torch.no_grad()
    def predict_action_and_risk(self, seq_np) -> tuple[int, int]:
        logits_action, logits_risk = self.predict_logits(seq_np)
        a = int(logits_action.argmax(dim=-1).item())
        r = int(logits_risk.argmax(dim=-1).item())
        return a, r
