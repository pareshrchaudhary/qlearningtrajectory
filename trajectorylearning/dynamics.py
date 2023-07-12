import numpy as np
import math
import torch

rad2deg = 180.0 / np.pi
deg2rad = np.pi / 180.0
dt = 0.02

class CarEnvironment:
    def __init__(self, dt=0.02):
        self.L = 1  # Length of the car
        self.dt = dt  # Time step
        self.done = False

    def car_dynamics(self, state, action):
        assert len(state) == 3, "state must be a tensor of shape (3,)"
        assert len(action) == 2, "action must be a tensor of shape (2,)"

        x, y, theta = state
        action = np.clip(action, 0.0, 1.0)
        velocity, delta = action

        xnew = x + torch.cos(theta) * velocity * self.dt
        ynew = y + torch.sin(theta) * velocity * self.dt
        thetanew = theta + (velocity / self.L) * torch.tan(delta) * self.dt
        thetanew = thetanew % (2.0 * math.pi)
        new_state = torch.tensor([xnew, ynew, thetanew])
        return new_state

    def find_nearest(self, target, current):
        assert target.shape == torch.Size([65, 3]),  "target must be a (65,3) tensor"
        assert len(current) == 3, "state must be a tensor of shape (3,)"

        lookahead = 2
        x, y, theta = current

        inter_x = (target[:-lookahead-1, 0] - x)**2
        inter_y = (target[:-lookahead-1, 1] - y)**2
        dist = torch.sqrt(inter_x + inter_y)
        minindex = torch.argmin(dist)
        return minindex

    def calculate_deviation(self, mi, target, current):
        assert mi >= 0, "mi must be a non-negative integer"
        assert target.shape == torch.Size([65, 3]),  "target must be a (65,3) tensor"

        x, y, theta = current
        nearest_trajectory = torch.stack((target[mi, 0] - x, target[mi, 1] - y), dim=0)

        car_xvec = torch.cos((math.pi/2.0) + theta)
        car_yvec = torch.sin((math.pi/2.0) + theta)
        carLeftvec = torch.stack((car_xvec, car_yvec), dim=0)

        error = -torch.dot(nearest_trajectory, carLeftvec)
        return error

    def calculate_reward(self, state, target):
        assert len(state) == 3, "state must be a tensor of shape (3,)"
        assert target.shape == torch.Size([65, 3]),  "target must be a (65,3) tensor"

        nearest_index = self.find_nearest(target, state)
        deviation_error = self.calculate_deviation(nearest_index, target, state)

        if deviation_error >= 0:
            reward = 1.0
        else:
            reward = -1.0

        return reward

    def step(self, state, action, target):
      action = np.clip(action, 0.0, 1.0)
      next_state = self.car_dynamics(state, action)
      reward = self.calculate_reward(state, target)

      index = self.find_nearest(target, state) # Termination
      if (index == 65):
        self.done = True

      return next_state, reward, self.done

if (__name__ == '__main__'):
  # Create an instance of CarEnvironment
  env = CarEnvironment()

  # Define a state, action, and target tensors
  state = torch.tensor([0.0, 0.0, 0.0])
  action = torch.tensor([1.0, 0.1])
  target = torch.randn(65, 3)

  # Calculate the reward
  reward = env.calculate_reward(state, target)
  print("Reward:", reward)

  # Perform a step in the environment
  next_state, reward, done = env.step(state, action, target)
  print("Next state:", next_state)
  print("Reward:", reward)
  print("Done:", done)