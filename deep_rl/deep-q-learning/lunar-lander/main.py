import gymnasium as gym
import torch
from IPython.display import Video
import matplotlib.pyplot as plt

from record_game import record_game
from train import trainer


env = gym.make("LunarLander-v3", render_mode="rgb_array")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)
agent, log = trainer(env, num_games=500,device=device)

record_game(env, agent, prefix="q_learning_unstable")
Video("save_videos/q_learning_unstable-episode-0.mp4", embed=True)


plt.plot(log["scores"], label="Scores")
plt.plot(log["running_avg_scores"], label="Moving Avg")
plt.title("Lunar Lander Scores per Game")
plt.xlabel("Game Number")
plt.ylabel("Score")
plt.legend()
plt.show()
