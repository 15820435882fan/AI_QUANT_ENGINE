V30 Step 1: Teacher + Dataset Builder

Files in this package:

- v30_teacher_strategies.py
    * Defines TeacherConfig and TeacherState.
    * teacher_trend_follow(df) takes a 1h OHLCV DataFrame and:
        - Computes MA20/MA60, ATR14, trend_slope, volatility.
        - Generates:
            action_teacher (0=HOLD,1=LONG,2=SHORT,3=CLOSE)
            risk_level (0=low,1=medium,2=high)
            position_side_teacher
            entry_price_teacher

- v30_dataset_builder.py
    * Uses LocalDataEngineV22_9 (or local_data_engine.load_local_kline) to load data.
    * Runs teacher_trend_follow() to get teacher actions.
    * Builds features (close_norm, returns, MA ratios, atr_ratio, vol_24, trend_slope).
    * Converts to sliding sequences of length seq_len.
    * Splits chronologically into train/valid/test and saves as:
        datasets/v30_<symbol>_<timeframe>_train.npz
        datasets/v30_<symbol>_<timeframe>_valid.npz
        datasets/v30_<symbol>_<timeframe>_test.npz

Basic usage (inside your project venv):

1) Put these files next to your existing local_data_engine_v22_9.py

2) Build BTCUSDT 1h dataset for last ~8 years (2920 days):
    python v30_dataset_builder.py --symbol BTCUSDT --timeframe 1h --days 2920 --seq-len 128

3) Check 'datasets' directory for generated .npz files,
   then we will build v30_model.py and v30_train_supervised.py to train the first V30 brain.
