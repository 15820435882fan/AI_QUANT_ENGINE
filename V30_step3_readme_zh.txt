V30 Step3：Transformer 模型接入回测环境（行为 + 收益检验）

本步骤的目标：
1）不再只看“分类准确率”，而是看：
   - 模型在完整历史 K 线上的交易行为
   - 最终收益、回撤、交易次数、胜率
2）与 Teacher 老师策略做一个直接对比：
   - 看看 AI 学徒是否已经接近甚至超越老师

本压缩包包含 2 个核心文件：

--------------------------------------------------
1. v30_policy_wrapper.py
--------------------------------------------------
功能：封装已经训练好的 V30 Transformer 模型，提供简单预测接口。

主要内容：
- 类：V30TransformerPolicy
  - 初始化：
        policy = V30TransformerPolicy(model_path="models_v30/v30_transformer_BTCUSDT_1h_best.pt")
  - 关键方法：
        predict_action(seq_np) -> int
        predict_action_and_risk(seq_np) -> (int, int)

  其中：
  - seq_np 形状为 [seq_len, feature_dim] 的 numpy 数组
  - 返回的动作编码：
        0 = HOLD
        1 = LONG（开/持多）
        2 = SHORT（开/持空）
        3 = CLOSE（平仓）

注意：
- 该策略内部会自动根据 checkpoint 中的 seq_len、feature_dim 构造模型。
- device 为 auto（优先使用 GPU），也可以改为 cpu。


--------------------------------------------------
2. v30_eval_in_env.py
--------------------------------------------------
说明：这里虽然文件名叫 in_env，但实际是自带一个轻量级回测引擎，
直接在原始 K 线 + 老师同一套特征上跑回测。

这样做的好处：
- 和 V30 训练数据完全一致（同一套特征 / 老师规则）
- 避免和 V22 环境的 state_dim / 特征不一致导致偏差
- 方便专注检验 “模型行为 + 收益表现”

主要流程：
1）加载原始 K 线（调用 v30_dataset_builder.load_ohlc）
2）运行 teacher_trend_follow，得到老师动作 + ATR 等指标
3）调用 build_features，构建与训练时完全一致的特征
4）用 V30 Transformer 模型，对每个时间点生成一个动作序列
5）用内置简易回测引擎，对比：
    - V30 模型动作 → 回测结果
    - Teacher 老师动作 → 回测结果

回测规则（简化版）：
- 初始资金：默认 10000 USDT
- 动作含义：
    0: HOLD       → 不操作
    1: 开/持多    → 空仓时全仓开多；持多时继续持有
    2: 开/持空    → 空仓时全仓开空；持空时继续持有
    3: CLOSE      → 平掉当前仓位，回到空仓
- 多头：
    - 开仓：units = equity / price
    - 每根 bar 用最新价格做市值：equity = units * price
- 空头：
    - 开仓：units = equity / price
    - 每根 bar 的权益：equity = entry_equity + (entry_price - price) * units
- 如最后一根 K 线仍在持仓，会自动按最后价平仓。

输出指标：
- 最终权益 final_equity
- 总收益率
- 最大回撤 max_drawdown
- 交易次数 num_trades
- 胜率 win_rate

命令示例（建议在项目根目录运行）：

    python v30_eval_in_env.py --symbol BTCUSDT --timeframe 1h --days 2920 \
        --model-path models_v30/v30_transformer_BTCUSDT_1h_best.pt --seq-len 128 --initial-equity 10000

参数说明：
- --symbol          交易对，例如 BTCUSDT
- --timeframe       周期，例如 1h
- --days            回测回看多少天历史（需和你数据源匹配，例：2920 ≈ 8 年）
- --model-path      V30 Transformer 模型路径
                    为空时默认使用 models_v30/v30_transformer_<symbol>_<timeframe>_best.pt
- --seq-len         序列长度，需与训练时一致（当前为 128）
- --initial-equity  初始资金，默认 10000

运行后，会打印两组结果：
- V30 Transformer 模型 回测结果
- Teacher 老师策略 回测结果

示例输出结构：

    ===== V30 Transformer 模型 回测结果 =====
    最终权益: 12345.67
    总收益率: 23.46%
    最大回撤: 12.34%
    交易次数: 120
    胜率: 58.33%
    ===========================
    ===== Teacher 老师策略 回测结果 =====
    最终权益: 11890.12
    总收益率: 18.90%
    最大回撤: 10.85%
    交易次数: 105
    胜率: 55.24%
    ===========================

解读方式建议：
1）如果 V30 最终权益 ≥ 老师策略，说明：
   - 模型至少没有比老师更差，已经学会了绝大多数老师逻辑；
   - 可能还在部分区间上做出了“改进老师”的行为。

2）如果 V30 最大回撤 < 老师策略，说明：
   - 模型在风险控制上更稳健（例如更少在高 ATR / 高波动区间重仓）。

3）如果 V30 胜率略低但收益更高，说明：
   - 模型可能更偏向“吃大趋势”，而不是频繁小打小闹。

后续可以进一步扩展：
- 画出两条 equity 曲线
- 对具体某一段时间，打印 K 线 + 模型动作 + 老师动作
- 做出“行为分析报告”，看 AI 在什么行情下最聪明，哪里还需要强化学习微调。
