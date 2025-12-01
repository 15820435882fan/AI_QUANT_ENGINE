from trading_env_v22_2 import TradingEnvV22MultiTF

env = TradingEnvV22MultiTF(symbol="BTCUSDT", days=365)
obs = env.reset()
print("初始 state 维度:", obs.shape)
print("state 示例前 10 项:", obs[:10])

done = False
step = 0
total_reward = 0.0

while not done and step < 300:
    action = env.action_space_sample()  # 暂时随机，下步会用 RL 模型替换
    obs, reward, done, info = env.step(action)
    total_reward += reward
    step += 1

print("总步数:", step, "总reward:", total_reward, "最终权益:", info["equity"])
