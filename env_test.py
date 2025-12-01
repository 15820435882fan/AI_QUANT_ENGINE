from trading_env_v22_1 import TradingEnvV22

env = TradingEnvV22(symbol="BTCUSDT", days=365)
obs = env.reset()
print("初始 obs:", obs)

done = False
step = 0
while not done and step < 200:
    action = env.action_space_sample()  # 暂时随机，后面换成 RL 模型输出
    obs, reward, done, info = env.step(action)
    print(step, "action=", action, "reward=", reward, "equity=", info["equity"])
    step += 1
