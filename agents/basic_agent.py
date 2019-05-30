import numpy as np
from task import Task
import random

class BasicAgent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

		#scoring measures
        self.best_score = -np.inf
        self.score = 0

    def reset_episode(self):
        self.score = 0

        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        self.score += reward

        if(done and self.score > self.best_score) : 
            self.best_score = self.score
       
        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        new_thrust = random.gauss(450., 25.)
        return [new_thrust + random.gauss(0., 1.) for x in range(4)]

    