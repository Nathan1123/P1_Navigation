[//]: # (Image References)

[image1]: https://i.imgur.com/SQxhG23.png "Results Chart"

# Project 1: Navigation

Nathan Goedeke

Deep Reinforcement Learning

March 28, 2022

### Algorithm Summary

The approach to this project uses the Deep Q-Network Reinforcement Learning Algorithm. Reinforcement Learning creates a table of values Q(S,A) that estimates the expected payout by taking a given action A while in state S. With each new state, the agent selects the action giving the largest expected payout from the Q-table. The function giving best action for each state is then called the policy. After recieving a new state and reward from this action, the agent updates the Q-table using the information (S,A,R,S',A'). Thus, the reinforcement algorithm alternates in a cycle of three steps:

1. Create a new policy Pi(S) from the Q-table by selecting the best action A for each state.

2. Select new action A' from the policy, and obtain reward R and new state S'

3. Update the Q-table at entry Q(A,S) using (S,A,R,S',A')

Creating a policy using the maximum expected payout is a greedy approach. Alternatively, a probability factor epsilon is used to balance between exploitation and exploration, such that a value of eps=0 is a greedy approach while eps=1 selects actions completley at random. 

A Deep Q-Network augments this above approach by using a neural network to approximate the function Q(S,A). The information of (S,A,R,S',A') is used to update the weights of this network at a given rate. 

An experience replay buffer is used to record previous tuples of data (S,A,R,S',A'), which is pulled from randomly in order to update the network from previous runs. 

### Parameters

These are all the parameters used to generate this model. These first four parameters are already given by the nature of the problem:

* State Size - The number of dimensions in the environment (37)
* Action Size - The number of possible actions at any given state (4)
* Seed - Pseudo- Random seed used in the learning model. Set to zero here for true randomness
* Score Window - Number of episodes to average for a total score (100)

These remaining parameters are determined by experimentation and research:

* Buffer size - Number of tuples (S,A,R,S',A') stored in the experience replay buffer
* Batch size - Size of batch data in the neural network
* Gamma - The discount factor, which determines how much information diminishes over multiple runs. A value of 1 remembers everything and a value of 0 remembers nothing.
* Tau - Controls how much information is balanced between the Target Q-Network and the Local Q-Network
* Learning Rate - How sensitive the neural network is to updating weights
* Update Every - How many actions are taken before updating the network weights
* N Episodes - Total number of episodes to run
* Max T - Maximum number of actions to take before ending an episode. Otherwise, the episode will end when the environment returns Done=True
* Epsilon - Value to balance exploration and exploitation, where a value of zero will be a greedy approach. In this project, epsilon starts at a value of one and diminishes at a rate of EPS_Decay, stopping at a minimum value EPS_End. 

### Results

The agent successfully trained after 1800 episodes to collect an average reward of 15-16 over 100 episodes. The results per 100 episodes and chart are displayed below:

Episode 100     Average Score: 0.54<br>
Episode 200     Average Score: 4.03<br>
Episode 300     Average Score: 7.45<br>
Episode 400     Average Score: 9.42<br>
Episode 500     Average Score: 12.40<br>
Episode 600     Average Score: 13.41<br>
Episode 700     Average Score: 14.43<br>
Episode 800     Average Score: 15.08<br>
Episode 900     Average Score: 15.54<br>
Episode 1000     Average Score: 14.54<br>
Episode 1100     Average Score: 15.77<br>
Episode 1200     Average Score: 14.88<br>
Episode 1300     Average Score: 15.42<br>
Episode 1400     Average Score: 15.84<br>
Episode 1500     Average Score: 15.36<br>
Episode 1600     Average Score: 15.53<br>
Episode 1700     Average Score: 15.67<br>
Episode 1800     Average Score: 14.73

![Results Chart][image1]

### Future Work

* Better performance could possibly be achieved by fine tuning other parameters. It seems likely the agent could get as high as 18
* Implementing a priority factor to the experience replay buffer could make better use of the replay system