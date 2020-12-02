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

# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.layer_3 = torch.nn.Linear(in_features=100, out_features=100)
        self.layer_4 = torch.nn.Linear(in_features=100, out_features=100)
        self.layer_5 = torch.nn.Linear(in_features=100, out_features=100)
        self.layer_6 = torch.nn.Linear(in_features=100, out_features=100)
        self.layer_7 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        layer_4_output = torch.nn.functional.relu(self.layer_4(layer_3_output))
        layer_5_output = torch.nn.functional.relu(self.layer_5(layer_4_output))
        layer_6_output = torch.nn.functional.relu(self.layer_6(layer_5_output))
        layer_7_output = torch.nn.functional.relu(self.layer_7(layer_6_output))
        output = self.output_layer(layer_7_output)
        return output

# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=6)
        # Create a Target-network, which predicts the q-value for a particular, and helps us save the earlier model
        self.target_network = Network(input_dimension=2, output_dimension=6)
        # Define the replace buffer
        self.buffer = PrioritisedReplayBuffer(size=BUFFER_SIZE, weight_alpha=WEIGHT_ALPHA)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        # Load the Q-network to the Target Network
        self.target_network.load_state_dict(self.q_network.state_dict())

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, minibatch_size):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()

        # Get a minibatch
        minibatch, indices = self.buffer.sample_minibatch(minibatch_size)
        # Calculate the loss for this transition.
        loss = self._calculate_loss(minibatch, indices)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, minibatch, indices):
        # Define the tensors
        state_tensor = torch.tensor(minibatch[0], dtype=torch.float32) # state tensor
        prime_state_tensor = torch.tensor(minibatch[3], dtype=torch.float32) # state prime tensor
        actions_tensor = torch.tensor(minibatch[1], dtype=torch.int64) # action tensor
        reward_tensor = torch.tensor(minibatch[2], dtype=torch.float32) # reward tensor

        # Get the Q-function values
        q_values = self.q_network.forward(state_tensor)
        q_values_prime  = self.q_network.forward(prime_state_tensor)

        # Get the Q-function values with the Target Network and the next state
        target_values_prime = self.target_network(prime_state_tensor) # Q_hat(S',a)
        target_values_prime_best_action = torch.argmax(q_values_prime, 1) # argmax_a Q_hat(S', a)

        # max_q_next_state = torch.max(q_values_prime, 1)[0].detach() # pick it across the actions tensor
        max_q_next_state = torch.gather(q_values_prime, 1, target_values_prime_best_action.unsqueeze(-1)).squeeze(-1)  # Q(S', argmax_a Q_hat(S, a))

        # Pick the Q values with the best action
        q_values_best_action = torch.gather(q_values, 1, actions_tensor.unsqueeze(-1)).squeeze(-1) # Q(S,A)
        
        # Find the prediction tensor
        # R + Q(S', argmax_a Q_hat(S, a))
        # Make the prediction
        prediction = torch.add(reward_tensor,max_q_next_state, alpha=0.9)

        # Update the weights of the batch
        weights = torch.abs(torch.subtract(prediction, q_values_best_action) + WEIGHT_EPSILON)
        self.buffer.update_weights(weights, indices)

        # Calculate loss
        loss = torch.nn.MSELoss()(prediction, q_values_best_action)
        return loss


    def update_target_network(self):
        # Updates the Q-Network
        return self.target_network.load_state_dict(self.q_network.state_dict())

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

