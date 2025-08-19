import numpy as np
import pickle

class Agent:
    def __init__(self):
        self.is_high = True # Used to indicate if the energy is high or low
        
        # Free to change for experimenting
        self.r_wait = 1
        self.r_search = 3
        self.alpha = 0.5
        self.beta = 0.5
        
    def act(self):
        if self.is_high:
            pass

    