[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"

# Project 1: Navigation 

### Overview

In this project, an intelligent agent will be trained to navigate a large, square world and collect bananas (yes, bananas). 

![Trained Agent][image1]

The agent is rewarded +1 for each yellow banana and -1 for each blue banana. Thus, it is trained to maximize the amount of yellow bananas while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction. The agent uses this information to select the next best action.  Four discrete actions are available: forward, backward, left, and right.

The task is episodic, and in order to solve the environment, the trained agent must get an average score of +13 over 100 consecutive episodes.

### Prerequisites

The following instructions should be completed prior to running the code:

0. Ensure the following are already installed:
* Python 3.7 or higher
  * numpy, matplotlib, unityagents, torch
* Anaconda
  * Jupyter notebooks
  * Git

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. If running in **Windows**, ensure you have the "Build Tools for Visual Studio 2019" installed from this [site](https://visualstudio.microsoft.com/downloads/).  This [article](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30) may also be very helpful.  This was confirmed to work in Windows 10 Home.  

3. To install the base Gym library, use `pip install gym`. Supports Python 3.7, 3.8, 3.9 and 3.10 on Linux and macOS. 
	
4. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.  
    ```bash
    git clone https://github.com/udacity/deep-reinforcement-learning.git
    cd deep-reinforcement-learning/python
    pip install .
    ```

5. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.    
    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

6. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]

7. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

8. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### File descriptions

- model.py - defines neural network used to train the agent
- dqn_agent.py - defines agent, with functions to act, learn and remember previous states in a replay buffer
- Deep_Q_Learning.py - sets up an agent in an environment, trains it, and displays results
- Banana_Collecting_DQL.py - sets hyperparameters and uses Deep_Q_Learning to train an agent in the Banana collecting environment 
- Navigation.ipynb - Jupyter notebook to run the training and simulation
- Banana_Collecting_weights.pth - output file containing saved weights of the trained model

### Instructions

The following instructions will run the environment simulation, train the agent and display results:

1. Open Anaconda and navigate to the folder containing Navigation.ipynb. Run the following commands to open the notebook:
    ```bash
    conda activate drlnd
	jupyter notebook Navigation.ipynb
    ```
2. The usable code starts in block 5, and all other blocks can be ignored. The hyperparameters can be altered as the user sees fit.