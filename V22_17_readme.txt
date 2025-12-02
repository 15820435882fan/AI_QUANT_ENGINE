V22_17 Strong Trend Window Version

Files:
- trading_env_v22_17.py
    * Class: TradingEnvV22MultiTFV17
    * z-score window changed from 200 -> 100 to reduce warm-up and increase usable_len
    * Reward shaping logic kept from V22_16 (trend x3, momentum x2, stronger anti-trend and drawdown penalties)

- ai_trader_ppo_v22_17.py
    * VERSION_TAG = V22_17
    * rollout / batch_size = 8192, total_steps default = 150k, entropy_coef = 0.15
    * symbol -> days auto mapping:
         BTCUSDT -> 500
         ETHUSDT -> 540
         BNBUSDT -> 500
         SOLUSDT -> 360
      If --days > 0 is provided, it overrides the mapping.

Recommended commands:
- BTC:
    python ai_trader_ppo_v22_17.py --symbol BTCUSDT

- ETH:
    python ai_trader_ppo_v22_17.py --symbol ETHUSDT

- Override days manually:
    python ai_trader_ppo_v22_17.py --symbol BTCUSDT --days 720 --total-steps 200000
