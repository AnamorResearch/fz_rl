import numpy as np

from models import GeneratorModel


class RolloutBuffer():
    def __init__(self):
        self.actions = []
        self.states = []
        self.next_states = []
        self.costs = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def push(self, action, state, next_state, logprobs, reward, is_terminal):
        self.actions.append(action)
        self.states.append(state)
        self.next_states.append(next_state)
        self.logprobs.append(logprobs)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)

    def clear(self):
        self.actions = []
        self.states = []
        self.next_states = []
        self.costs = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def to_array(self):
        self.actions = np.array(self.actions)
        self.states = np.array(self.states)
        self.next_states = np.array(self.next_states)
        self.costs = np.array(self.costs)
        self.logprobs = np.array(self.logprobs)
        self.rewards = np.array(self.rewards)
        self.is_terminals = np.array(self.is_terminals)

    def std_rewards(self):
        self.rewards = np.array(self.rewards)
        self.rewards = (self.rewards - self.rewards.mean()) / (self.rewards.std() + 1e-7)

    def cum_rewards(self, gamma, v_func, next_v_func, _lambda=1.0):
        size = len(self.actions)
        gae = np.zeros((size, 1))
        deltas = self.rewards.reshape((-1, 1)) + gamma * next_v_func - v_func
        gae[-1] = deltas[-1]
        for t in reversed(range(size - 1)):
            gae[t] = deltas[t] + (1 - self.is_terminals[t + 1]) * (gamma * _lambda) * gae[t + 1]
        return gae


class Agent():
    def __init__(self, input_shape, action_space_dim, max_epoch, batch_num):
        self.max_epoch = max_epoch
        self.input_shape = input_shape
        self.action_space = action_space_dim
        self.buffer = RolloutBuffer()
        self.generator = GeneratorModel(input_shape=input_shape, action_space_dim=action_space_dim, batch=batch_num)

    def act(self, state):
        state = np.array([state])
        action, action_prob = self.generator.act(state=state)

        return action, action_prob
