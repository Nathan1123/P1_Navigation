from Deep_Q_Learning import DQ_Learner

# --------------------------------------------------------
#
# Nathan Goedeke
# Deep Reinforcement Learning - Project 1 (Navigation)
# March 28, 2022
#
# Description: This script will initialize an environment 
#              and agent that will be trained to collect 
#              yellow bananas and avoid blue bananas. 
#              It is trained over 1800 episodes to achieve 
#              an average score of 16 per 100 episodes.
#
# Usage: Set the hyperparameters below to fine tune the 
#        agent training, then execute the file in a python 
#        interpreter. Alternatively, the Jupyter Notebook 
#        Navigation.ipynb performs the same actions as this 
#        file. For more information, see the README file.
#
# ---------------------------------------------------------

# --- SET PARAMETERS HERE ---
env_name='Banana_Windows_x86_64\Banana.exe' # Path to executable
state_size=37 # Number of dimensions in environment
action_size=4 # Number of possible actions
seed=0        # Random seed for learning model. Zero for true randomness

buffer_size = int(1e5)  # Replay buffer size
batch_size = 64         # Minibatch size
gamma = 0.99            # Discount factor
tau = 1e-3              # For soft update of target parameters
lr = 5e-4               # Learning rate 
update_every = 4        # How often to update the network

n_episodes=1800  # Total number of episodes
max_t=5000       # Number of actions per episode
eps_start=1.0    # Initial value for epsilon [probability of taking greedy action]
eps_end=0.01     # Minimum epsilon value
eps_decay=0.995  # Rate of diminishing epsilon value
score_window=100 # Number of episodes to average results
model_name='Banana_Collecting_weights' # Name to save fully trained model

x_axis='Episode #' # Labels of results graph
y_axis='Score'
# ---------------------------

# Set up Unity environment
agent_params = [buffer_size, batch_size, gamma, tau, lr, update_every]
DQL = DQ_Learner(env_name, state_size, action_size, seed, agent_params)
# Train for n episodes
DQL.deep_q_learn(n_episodes, max_t, eps_start, eps_end, eps_decay, model_name, score_window)
input("Training complete. Press any key to display results")
# Display training results
DQL.display(x_axis, y_axis)
# Close environment
DQL.terminate()