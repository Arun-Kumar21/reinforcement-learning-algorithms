import numpy as np
from agent import Agent


def trainer(env, 
            num_games=500,
            min_reward=200,
            game_tolerance=10,
            max_memories=100_000,
            gamma=0.99,
            lr=0.001,
            batch_size=64,
            input_state_features=8,
            num_actions=4,
            hidden_features=128,
            epsilon=1.0,
            epsilon_decay=0.999,
            min_epsilon=0.05,
            log_freq=50,
            running_avg_steps=25,
            update_target_freq=100,
            device="cpu"):

    agent=Agent(max_memories=max_memories,
                gamma=gamma,
                lr=lr, 
                input_state_features=input_state_features,
                num_actions=num_actions,
                hidden_features=hidden_features,
                epsilon=epsilon,
                epsilon_decay=epsilon_decay,
                min_epsilon=min_epsilon,
                device=device)

    ending_tol=0

    log={"scores": [],
         "running_avg_scores": []}

    for i in range(num_games):

        score = 0
        step = 0

        state, _ = env.reset()
        done = False

        while not done:
            action = agent.select_action(state)

            next_state, reward, terminal, truncated, _ = env.step(action)
            done  = terminal or truncated

            score += reward

            agent.replay_buffer.add_memories(state, next_state, action, reward, done)

            agent.train_step(batch_size)

            # Update target Network after some steps
            if step%update_target_freq==0:
                agent.update_target_network()
                # print("Target network updated")

            step+=1

            state = next_state

        log["scores"].append(score)
        running_avg_scores = np.mean(log["scores"][-running_avg_steps:])
        log["running_avg_scores"].append(running_avg_scores)
        
        if i%log_freq==0:
            print(f"Game: {i} | Score: {score} | Moving avg scores: {running_avg_scores} | Epsilon: {agent.epsilon}")

        if score >= min_reward:
            ending_tol += 1

            if ending_tol == game_tolerance:
                break
 
        else:
            ending_tol=0 


    print("Complete Training")

    return agent, log

