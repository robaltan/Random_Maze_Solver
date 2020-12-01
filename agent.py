# Implements Reinforcement Learning methods to solve a random maze

import numpy as np
import torch 
import collections

# Define the parameters for Deep-Q Learning
EPSILON = 0.9999
EPSILON_DECAY = 0.9999
EPISODE_LENGTH = 600
K = 5 # Episode interval to update the Target Network
MINIBATCH_SIZE = 100
BUFFER_SIZE = 5000
EPSILON_GREEDY_START_STEP = 3000
CONTINUOUS_ACTION = 0.02
WEIGHT_EPSILON = 0.0005
WEIGHT_ALPHA = 0.7
EPSILON_CONTROL = True
DISTANCE_ALPHA = 3.0
RANDOM_ACTIONS_END_EPISODE = 250
GREEDY_STEPS = 100

class PrioritisedReplayBuffer:
    # Prioritised Replay Buffer that allows the agent to learn from a minibatch
    def __init__(self, weight_alpha, size=5000):
        # Initialize the replay buffer
        self.buffer = collections.deque(maxlen=size)
        self.weights = collections.deque(maxlen=size)
        self.size = size
        self.weight_alpha = weight_alpha

    def calculate_transition_weight(self, transition, dqn):
        # Extract (S, A, R, S')
        # Use only if you haven't started using minibatch
        (state, action, reward, prime_state) = transition
        state_tensor = torch.tensor(state, dtype=torch.float32)
        prime_state_tensor = torch.tensor(prime_state, dtype=torch.float32)

        # Get the Q-function for the given state and action
        q_value = dqn.q_network.forward(state_tensor)[action]
        # Get the Q-function values with the Q-Network and the next state
        q_values_prime = dqn.q_network(prime_state_tensor)

        # Calculate prediction
        prediction = reward + q_values_prime.max()

        # Calculate weight
        weight = abs(prediction - q_value).detach().numpy() + WEIGHT_EPSILON
        return weight

    def add_transition(self, transition, dqn):
        # Adds the transition to the buffer
        if self.buffer == self.size * 2:
            self.buffer.popleft()
            self.weights.popleft()
        self.buffer.append(transition)
        if (len(self.buffer)  == 1):
            self.weights.append(self.calculate_transition_weight(transition, dqn))
        else:
            # Assign the largest weight possible because this is a new weight
            self.weights.append(max(self.weights))

    def sample_minibatch(self, minibatch_size):
        # Sample a minibatch without any replacements
        # Calculate the probability vector
        weights = np.power(self.weights, WEIGHT_ALPHA)
        probabilities = np.array(weights / np.sum(weights))
        indices = np.random.choice(len(self.buffer), minibatch_size, replace=False, p=probabilities)
        states, actions, rewards, state_prime = zip(*[self.buffer[i] for i in indices])
        return (np.array(states), np.array(actions), np.array(rewards,dtype=np.float32), np.array(state_prime)), indices
    
    def get_size(self):
        # Return the size of the replay buffer
        return len(self.buffer)

    def update_weights(self, weights, indices):
        # Updates the weights of the buffer with new 
        new_weights = weights.detach().numpy()
        for i in range(new_weights.shape[0]):
            self.weights[indices[i]] = new_weights[i]


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = EPISODE_LENGTH
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # Define epsilon
        self.epsilon = EPSILON
        # Define epsilon decay
        self.epsilon_decay = EPSILON_DECAY
        # Define episode interval to update the target network
        self.k = K * EPISODE_LENGTH
        # Define minibatch size
        self.minibatch_size = MINIBATCH_SIZE
        # Define epsilon greedy start step
        self.epsilon_greedy_start_step = EPSILON_GREEDY_START_STEP
        # Define small epsilon for Prioritised Experience Replay
        self.weight_epsilon = WEIGHT_EPSILON
        # Create a DQN (Deep Q-Network)
        self.dqn = DQN()
        # Create EARLY STOPPING boolean
        self.early_stopping = False
    
    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        # If the modulo of numbers of steps taken and episode length is 0, then you start a new episode
        if self.num_steps_taken % self.episode_length == 0:
            return True
        else:
            return False
    
    # Function to convert a discrete action to a continuous action
    def _discrete_action_to_continuous(self, discrete_action):
        # Action space includes going to left, right, up, down, diagonal top right, and diagonal below right
        if discrete_action == 0:
            continuous_action = np.array([0,CONTINUOUS_ACTION], dtype=np.float32)
        elif discrete_action == 1:
            continuous_action = np.array([0,-CONTINUOUS_ACTION], dtype=np.float32)
        elif discrete_action == 2:
            continuous_action = np.array([CONTINUOUS_ACTION, 0], dtype=np.float32)
        elif discrete_action == 3:
            continuous_action = np.array([CONTINUOUS_ACTION/2.0, CONTINUOUS_ACTION/2.0], dtype=np.float32)
        elif discrete_action == 4:
            continuous_action = np.array([-CONTINUOUS_ACTION, 0], dtype=np.float32)
        else:
            continuous_action = np.array([CONTINUOUS_ACTION/2.0, -CONTINUOUS_ACTION/2.0], dtype=np.float32)

        return continuous_action

    # Function to convert a continuous action to a discrete action
    def _continuous_action_to_discrete(self, continuous_action):
        if np.array_equal(continuous_action, np.array([0,CONTINUOUS_ACTION], dtype=np.float32)): # x,y
            return 0
        elif np.array_equal(continuous_action, np.array([0,-CONTINUOUS_ACTION], dtype=np.float32)):
            return 1
        elif np.array_equal(continuous_action, np.array([CONTINUOUS_ACTION, 0], dtype=np.float32)):
            return 2
        elif np.array_equal(continuous_action, np.array([CONTINUOUS_ACTION/2.0, CONTINUOUS_ACTION/2.0], dtype=np.float32)):
            return 3
        elif np.array_equal(continuous_action, np.array([-CONTINUOUS_ACTION, 0], dtype=np.float32)):
            return 4
        else:
            return 5
    

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Here, the action is random, but you can change this
        if self.num_steps_taken < self.epsilon_greedy_start_step or ((self.num_steps_taken + RANDOM_ACTIONS_END_EPISODE) % self.episode_length >= 0 and (self.num_steps_taken + RANDOM_ACTIONS_END_EPISODE) % self.episode_length < RANDOM_ACTIONS_END_EPISODE):
            # Explore the environment randomly
            action = self._discrete_action_to_continuous(np.random.choice([0,1,2,3,4,5]))
        elif self.num_steps_taken % self.episode_length < GREEDY_STEPS:
            action = self.get_greedy_action(state)
        else:
            # Take action with the epsilon greedy policy
            action = self.get_epsilon_greedy_action(state)
    
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = action
        return action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        reward = (1 - distance_to_goal)**DISTANCE_ALPHA
        # Create a transition
        if (self.num_steps_taken % self.episode_length < GREEDY_STEPS and self.num_steps_taken % self.episode_length > 0) and distance_to_goal < 0.03:
            # print("EARLY STOPPING @ {} with distance to goal of {}".format(self.num_steps_taken % self.episode_length, distance_to_goal))
            self.early_stopping = True
        
        transition = (self.state, self._continuous_action_to_discrete(self.action), reward, next_state)
        if self.early_stopping == False:
            # Never want to train anymore because your greedy policy converge
            # Perform 100 greedy steps as to test your Q-network
            if (self.state == next_state).all():
                    reward = 0
                    
            if self.num_steps_taken % self.episode_length > GREEDY_STEPS:
                # If you are taking greedy policy
                self.dqn.buffer.add_transition(transition, self.dqn)

                # Whenever we start epsilon greedy, start training the q_network
                if self.dqn.buffer.get_size() >= self.epsilon_greedy_start_step:
                    self.dqn.train_q_network(self.minibatch_size)
                    
                # Check whether the Target Network needs to be updated
                if self.dqn.buffer.get_size() >= self.epsilon_greedy_start_step and (self.num_steps_taken + 1) % self.k == 0:
                    self.dqn.update_target_network()

    # Function that picks the epsilon greedy action
    def get_epsilon_greedy_action(self, state):
        # Define your state tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # Pick the best action
        best_action = np.argmax(self.dqn.q_network.forward(torch.unsqueeze(state_tensor, 0)).detach().numpy()[0])
        # Stop epsilon decay if epsilon is around 0.05
        if EPSILON_CONTROL:
            if self.epsilon > 0.05:
                self.epsilon *= self.epsilon_decay
        else:
            self.epsilon *= self.epsilon_decay

        # Choose an action with epsilon decay
        p = np.full(6,self.epsilon/6)
        p[best_action] = 1 - self.epsilon + self.epsilon/6
        return self._discrete_action_to_continuous(np.random.choice([0,1,2,3,4,5], 1, p=p)[0])
    
    # Function that picks the greedy action given the state
    def get_greedy_action(self, state):    
        # Define your state tensor
        state_tensor = torch.tensor(state, dtype=torch.float32)
        # Pick the best action
        best_action = np.argmax(self.dqn.q_network.forward(torch.unsqueeze(state_tensor, 0)).detach().numpy()[0])

        # Define the probability vector
        p = np.full(6,0)
        # Assign 1 to the best action to make sure that it gets selected
        p[best_action] = 1
        return self._discrete_action_to_continuous(np.random.choice([0,1,2,3,4,5], 1, p=p)[0])

