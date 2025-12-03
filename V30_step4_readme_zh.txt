V30 Step4：数据验证 + 多周期特征 + PPO 强化学习微调（B 路线升级版）

本步骤对应你选择的 C 路线要求：
  1）先验证数据源是否真实可靠（时间跨度、缺失、连续性）
  2）引入多周期特征（1h + 15m + 5m），让模型感知小级别结构
  3）在已有 Transformer 学徒基础上，用 PPO 做收益导向微调（真正学会“赚钱”）

本包包含 4 个核心文件：

--------------------------------------------------
1. v30_data_validator.py  —— 数据完整性验证
--------------------------------------------------
用途：
  在训练 / 回测 / 强化学习前，先检查本地 K 线数据是否连续、时间跨度是否足够。

使用示例：
  python v30_data_validator.py --symbol BTCUSDT --timeframe 1h --days 2920
  python v30_data_validator.py --symbol BTCUSDT --timeframe 15m --days 1095

输出内容包括：
  - 实际数据起止时间
  - 实际跨度天数
  - 实际 K 线数量
  - 理论 K 线数量（按周期推算）
  - 缺口数量 / 估算缺失 K 线数量
  - 结论：是否“基本连续且完整”

你可以用它来确认：
  - 1h 数据是否真的从 2017 年开始
  - 15m / 5m 是否有较多缺口
  - 是否适合用来做 8 年级别的训练与回测


--------------------------------------------------
2. v30_mtf_feature_builder.py  —— 多周期特征构造
--------------------------------------------------
目标：
  在每一根 1h K 线基础上，引入该小时内部 15m / 5m 的统计特征，
  让模型能够感知小级别结构（震荡、突破、放量等）。

主要流程：
  1）加载 1h K 线，调用 teacher_trend_follow() + build_features()
     得到：
       - 老师动作 action_teacher / risk_level
       - 1h 自身特征（close_norm, ret_1, ret_6, ...）

  2）加载 15m / 5m K 线：
       - 计算 ret_1 = close.pct_change()
       - 按 1H 重采样，得到：
           * m15_ret_1_mean / m15_ret_1_std / m15_high_max / m15_low_min
           * m5_ret_1_mean / m5_ret_1_std / m5_high_max / m5_low_min

  3）把 15m / 5m 聚合特征对齐到 1h 索引，拼到一起，形成 df_mtf。

返回：
  df_mtf, feat_cols

其中：
  - df_mtf.index 为 1h 时间
  - 包含：
      close, action_teacher, risk_level, 各种 1h 特征 + m15_* + m5_* 特征
  - feat_cols 是最终特征列列表


--------------------------------------------------
3. v30_rl_env.py  —— V30 强化学习环境（多周期）
--------------------------------------------------
这是一个简化版 RL 环境，用于 PPO 训练：

配置类：V30RLEnvConfig
  - symbol            交易对，例如 BTCUSDT
  - base_timeframe    "1h"
  - mid_timeframe     "15m"
  - fast_timeframe    "5m"
  - days              回看多少天作为训练数据
  - seq_len           状态序列长度（默认 128）
  - episode_steps     每个 episode 长度（默认 512）
  - initial_equity    初始资金（默认 10000）

环境行为：
  - reset():
      * 从历史中随机截取一段长度为 episode_steps 的区间
      * 重置仓位为空仓
      * 返回初始状态：shape = [seq_len, feature_dim]
  - step(action):
      * 动作定义：
          0 = HOLD
          1 = LONG（若空仓则全仓开多，持多则继续）
          2 = SHORT（若空仓则全仓开空，持空则继续）
          3 = CLOSE（平仓回空）
      * 每步先 mark-to-market，再执行动作
      * 奖励 = (equity_new - equity_old) / initial_equity
      * 返回 next_state, reward, done, info

说明：
  - 特征来自 v30_mtf_feature_builder.build_mtf_features()
  - 因为包含 1h + 15m + 5m 信息，模型可以感知“小级别震荡 / 假突破”。


--------------------------------------------------
4. v30_ppo_finetune.py  —— 使用 PPO 微调 Transformer（收益导向）
--------------------------------------------------
目标：
  在已有“老师模仿”基础上，让模型通过 PPO 学会“为了赚钱而优化行为”。

核心设计：
  - 模型：V30ActorCritic
      * Transformer 作为特征 backbone
      * policy_head 输出动作 logits（继续使用原来的 action_head）
      * value_head 输出状态价值
  - 初始化：
      * 从 v30_train_supervised 的 best.pt 中加载 Transformer 权重
      * 可选择：
          - 只训练 value_head + policy_head（稳妥）
          - 或连同 Transformer 一起微调（更激进）

  - PPOConfig:
      * total_steps     PPO 总训练步数（例如 200k）
      * rollout_steps   每次收集的步数（默认 1024）
      * gamma, lam      折扣与 GAE 参数
      * clip_ratio      PPO 截断系数
      * lr              学习率
      * vf_coef         value loss 系数
      * ent_coef        entropy 正则，鼓励探索
      * device          auto / cpu / cuda

使用示例：
  python v30_ppo_finetune.py --symbol BTCUSDT --timeframe 1h \
      --pretrained-path models_v30/v30_transformer_BTCUSDT_1h_best.pt \
      --total-steps 200000 --device auto

说明：
  - 默认只要你已经完成：
        1) V30 Step1：老师 + 数据集构建
        2) V30 Step2：Transformer 监督训练
        3) V30 Step3：初步回测（你已经完成）
    就可以直接运行 Step4 的 PPO 微调。
  - 微调完成后，会在 models_v30 下生成：
        v30_transformer_<symbol>_<timeframe>_ppo_finetuned.pt

后续：
  - 你可以用 v30_eval_in_env.py 把微调后的模型再跑一遍 8 年回测，
    直接比较：
       * 监督版 Transformer
       * PPO 微调版 Transformer
       * Teacher 老师策略
  - 真正看到“AI 学徒 → AI 交易员”的成长过程。


整体路线小结（B 路线升级版）：
  1）用 v30_data_validator 验证 1h / 15m / 5m 数据是否真实完整
  2）用 v30_mtf_feature_builder 构建多周期特征
  3）用 V30RLEnv + v30_ppo_finetune，在 1h+15m+5m 结构上做收益导向微调
  4）用 v30_eval_in_env 对 比：
       - 老师
       - 学徒（监督版）
       - 学徒（强化学习微调版）

这条路，就是你选的 B 路线（收益导向强化学习）在“数据真实 + 多周期结构”前提下的强化版实现。
