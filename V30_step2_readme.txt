V30 Step 2: Transformer model + supervised trainer

Files in this package:

1) v30_model.py
   - Defines:
       * V30TransformerConfig
       * PositionalEncoding (sinusoidal)
       * V30TransformerModel (TransformerEncoder-based)
       * build_v30_transformer() helper
   - Input:
       x: [batch_size, seq_len, feature_dim]
   - Outputs:
       logits_action: [batch_size, num_actions]
       logits_risk:   [batch_size, num_risks]

2) v30_train_supervised.py
   - Loads datasets from v30_dataset_builder.py:
       datasets/v30_<symbol>_<timeframe>_train.npz
       datasets/v30_<symbol>_<timeframe>_valid.npz
       datasets/v30_<symbol>_<timeframe>_test.npz
   - Training:
       * Supervised CrossEntropy on action + lambda_risk * CrossEntropy on risk
       * Optimizer: Adam
       * LR scheduler: ReduceLROnPlateau (monitors valid action accuracy)
       * Early stopping with patience=8
       * Saves best model to:
           models_v30/v30_transformer_<symbol>_<timeframe>_best.pt
   - Evaluation:
       * After training / early stopping, loads best model and prints:
           [V30 Test] acc_action=..., acc_risk=...

Basic usage (assuming BTCUSDT 1h dataset already built):

   python v30_train_supervised.py --symbol BTCUSDT --timeframe 1h \
       --data-dir datasets --model-dir models_v30 --epochs 30 --batch-size 128

You can adjust:
   --epochs, --batch-size, --lr, --lambda-risk, --device

Next step (Step 3) will be:
   - Wrap this model into a policy
   - Plug into TradingEnvV22MultiTFV17
   - Run backtests and compare with teacher and PPO baselines.
