import numpy as np
import pickle
import seaborn as sns

class Agent:
    def __init__(self):
        # Used to indicate if the energy is high or low
        self.is_high = True 

        # Training parameters
        self.epochs = 2000
        self.steps = 1000

        self.policy = {'high':{'search':5, 'wait':5},
                        'low':{'search':5, 'wait':5, 'recharge':5}
                    }

        # Free to change for experimenting
        self.r_wait = 1
        self.r_search = 3
        self.alpha = 0.2
        self.beta = 0.5
        self.epsilon = 0.1
        
    def act(self, action):
        if self.is_high:
            if action == 'search':
                if np.random.rand() < self.alpha:
                    self.is_high = False
                return self.r_search
            if action == 'wait':
                return self.r_wait
        if not self.is_high:
            if action == 'search':
                if np.random.rand() < self.beta:
                    return self.r_search
                else:
                    self.is_high = True
                    return -3
            if action == 'wait':
                return self.r_wait
            if action == 'recharge':
                self.is_high = True
                return 0
            
    def temporal_diff(self):
        if self.is_high:
            action = max(self.policy, key=self.policy.get())
            reward = self.act(action)
            self.policy['high'][action] += reward
        else:
            self.act(self.policy['low'].keys().values().max())
            
