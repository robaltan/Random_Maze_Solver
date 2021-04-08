# Random Maze Solver
A Reinforcement Learning agent, programmed in Python, that utilizes Double Q-Learning, 
Prioritized Experience Replay Buffer, and Early Stopping to solve a random maze. 

The agent is able to solve some of the mazes that are classified as *hard* meaning that there are multiple barriers that the agent needs to learn, and therefore does not get stuck in the local minimum. In the following picture, the agent was able to
learn to pass multiple wall barriers without any prior knowledge. It was also able to qualify for the competition round for the Reinforcement Learning graduate-level class that only selected 9% of the best agents.

![Agent Performance](https://github.com/robaltan/Random_Maze_Solver/blob/main/images/solved_maze.png)

## Deep Q-Learning

We used Deep Q-Learning to train the agent to understand the best policy given a location (x, y). Our agent used a neural network architecture that accepted a tuple of dimension 2, where the agent is currently at, and 
had 6 hidden layers of 100 edges in and out, and a final layer that predicted rewards for 6 different actions - right, up, bottom, left, diagonal up right, diagonal bottom right. 

![Deep Q-Learning](https://github.com/robaltan/Random_Maze_Solver/blob/main/images/deep_q_learning.png)

To make sure that learning is stable, we used a target network, which is a copy of the estimated value function that is held fixed to serve as a stable target for some number of steps. To make sure that the agent learns effectively, we utilized prioritized experience, which allows replaying important transitions more frequently. The agent remembered the actions it took, and the rewards it received, and it trained the network with a minibatch of size 100. For more details on the presentation, see [agent.py](https://github.com/robaltan/Random_Maze_Solver/blob/main/agent.py).


## Required Libraries
* PyTorch
* NumPy
* cv2 (*mainly for visualizations*)

## Objective

The objective of this project was to tinker with some RL algorithms on a very basic random maze environment. The following 
parameters are defined by the developer, but the user could modify them to see the effects of hyperparameter tuning. As 
we employ a really complex neural network with 7 layers, it made sense to perform early stopping; however, this might not
apply to some other RL environments. 

## Testing

To see how your agent performs, you should run the following command for the graphics package. In this way, you could see how your agent is performing in 600 seconds (relatively 10 minutes). It's useful to see the behavior of your agent.

```shell
python train_and_test.py
```

## What's Next?

Hyperparameter tuning and using different combinations of Reinforcement Learning methods is very important for the convergence of such algorithms. Therefore, designing a front-end interface where a user can define such variables would be helpful for visualization purposes, and deploying the front-end somewhere, so that everyone has access to it. Implementing methods from state-of-art papers might improve the performance of the agent drastically. 
