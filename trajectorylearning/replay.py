import random
import torch
from collections import deque

class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add_experience(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

    def sample_batch(self, batch_size):
        if len(self.buffer) < batch_size:
            return None  # Return None if buffer does not have enough samples

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)

if (__name__ == '__main__'):
  # Test buffer
  buffer = ReplayBuffer(buffer_size=1000)

  # Add experiences to the buffer
  state = torch.tensor([1.0, 2.0, 0.5])  # Example state
  action =  torch.tensor([0.1909, -0.0496])  # Example action
  reward = 0.1  # Example reward
  next_state = torch.tensor([1.2, 2.1, 0.6])  # Example next state
  done = False  # Example done flag
  buffer.add_experience(state, action, reward, next_state, done)

  # Sample a batch from the buffer
  batch_size = 1
  states, actions, rewards, next_states, dones = buffer.sample_batch(batch_size)

  # Check the length of the buffer
  buffer_length = len(buffer)

  print("States:", states)
  print("Actions:", actions)
  print("Rewards:", rewards)
  print("Next States:", next_states)
  print("Dones:", dones)
  print("Buffer Length:", buffer_length)