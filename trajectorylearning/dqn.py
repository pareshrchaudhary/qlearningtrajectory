import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        q_values = self.fc6(x)
        return q_values

    def initialize_weights(self):
      for module in self.modules():
        if isinstance(module, nn.Linear):
          nn.init.xavier_uniform_(module.weight.data)
          nn.init.constant_(module.bias.data, 0.0)

if (__name__ == '__main__'):
  #model test
  state_dim = 3
  action_dim = 2
  dqn = DQN(state_dim, action_dim)
  print(dqn)

