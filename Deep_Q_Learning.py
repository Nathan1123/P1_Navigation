from unityagents import UnityEnvironment
from dqn_agent import Agent
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch

class DQ_Learner:
    # Initialize environment and agent decision model
    def __init__(self, env_name, agent_states, agent_acts, agent_seed, agent_params):
        self.env = UnityEnvironment(file_name=env_name)
        self.agent = Agent(state_size=agent_states, action_size=agent_acts, seed=agent_seed, params=agent_params)
        self.scores = []
        print("NOTE: The environment will wait until deep_q_learn is run")
        
    # Function to train agent and record scores
    # This can be run any number of times before terminating the environment
    def deep_q_learn(self, n_episodes, max_t, eps_start, eps_end, eps_decay, model_name, score_window):
        scores = []
        brain_name = self.env.brain_names[0]
        eps = eps_start # Initialize epsilon
        # Loop per episode
        for i_episode in range(1, n_episodes+1):
            env_info = self.env.reset(train_mode=True)[brain_name] # Reset environment
            state = env_info.vector_observations[0] 
            score = 0
            # Actions per episode
            for t in range(max_t):
                action = self.agent.act(state, eps) # Take action
                env_info = self.env.step(int(action))[brain_name] # Update environment
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0] # Give reward
                done = env_info.local_done[0] # Check if episode is done
                self.agent.step(state, action, reward, next_state, done) # Train agent model from reward
                state = next_state
                score += reward # Cumulate reward
                # If episode ends, break
                if done:
                    break 
            # Statistics at end of episode
            scores.append(score)
            avg_score = np.mean(scores[-score_window:]) # Average score
            eps = max(eps_end, eps_decay*eps) # Update epsilon
            # Print results
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score), end="")
            if i_episode % score_window == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score))
        # Training complete, record scores and save model
        torch.save(self.agent.qnetwork_local.state_dict(), model_name+'.pth')
        self.scores = scores
        
    # Display score results
    def display(self, x_axis, y_axis):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(self.scores)), self.scores)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.show()
        
    # Close the environment
    def terminate(self):
        self.env.close()
        print("NOTE: Environment closed. No further training can be done.")