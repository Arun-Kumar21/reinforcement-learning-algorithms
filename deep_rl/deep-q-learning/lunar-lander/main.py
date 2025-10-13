import gymnasium as gym
import torch
import matplotlib.pyplot as plt

from record_game import record_game
from train import trainer


env = gym.make("LunarLander-v3", render_mode="rgb_array")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)
agent, log = trainer(
    env,
    num_games=500,
    batch_size=128,
    hidden_features=256,
    epsilon_decay=0.9995,
    gamma=0.99,
    lr=0.0005,
    device=device,
)

record_game(env, agent, prefix="q_learning_stable")

plt.plot(log["scores"], label="Scores")
plt.plot(log["running_avg_scores"], label="Moving Avg")
plt.title("Lunar Lander Scores per Game")
plt.xlabel("Game Number")
plt.ylabel("Score")
plt.legend()
plt.show()
