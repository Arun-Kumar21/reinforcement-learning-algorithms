import gymnasium as gym
import torch

def record_game(env, agent, prefix, device="cpu"):
    env = gym.wrappers.RecordVideo(env, 
                                   video_folder='./save_videos',
                                   video_length=0,
                                   disable_logger=True,
                                   name_prefix=prefix)
    
    done = False
    state, _ = env.reset()

    while not done:
        action = agent.inference(torch.tensor(state), device=device)

        new_state, reward, terminal, truncate, _ = env.step(action)
        done = terminal or truncate

        state = new_state

    env.close()
