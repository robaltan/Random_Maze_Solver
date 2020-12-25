# Random Maze Solver
A Reinforcement Learning agent, programmed in Python, that utilizes Double Q-Learning, 
Prioritized Experience Replay Buffer, and Early Stopping to solve a random maze. This agent was able to
solve the hardest mazeon Reinforcement Learning coursework @Imperial College London's Reinforcement
Learning class, which is solved by a tiny fraction of the class.

## Required Libraries
* PyTorch
* NumPy

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
