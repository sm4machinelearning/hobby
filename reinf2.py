# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 06:04:45 2021

@author: C64990
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time


class mygame:
    def __init__(
            self, 
            mesh
            ):
        self.mesh = mesh
        self.reset()
    
    def reset(self):
        self.position = np.where(self.mesh == 'start')[0]
        self.score = 0
        return self.position, self.score
    
    def play(self, position, action):
        mesh = self.mesh
        score = self.score
        
        # make a move
        print (position)
        new_position = position + action
        # ensure its inside boundary
        if new_position[0] >= 0 and new_position[0] < 4 and new_position[1] >= 0 and new_position[1] < 4:
            if mesh[new_position[0], new_position[1]] == 'lake':
                score = score - 10
            elif mesh[new_position[0], new_position[1]] == 'end':
                score = score + 100
            else:
                pass
        else:
            pass

        self.position = new_position
        self.score = score
        return self.position, self.score

########################
# define game

x, y = 4, 4
mesh = np.zeros((x, y), dtype='U5')
mesh[0, 0] = 'start'
mesh[1, 1] = 'lake'
mesh[1, 2] = 'lake'
mesh[2, 2] = 'lake'
mesh[3, 0] = 'lake'
mesh[3, 3] = 'end'

positions = []
for i in range(x):
    for j in range(y):
        positions.append([i,j])
actions = [[1,0],[-1,0],[0,1],[0,-1]]

n, num_trials = 0, 10
epsilon, gamma = 0.2, 1
qtable = pd.DataFrame(index=tuple(positions), columns=tuple(actions))
########################


########################
# run game
game = mygame(mesh)
while n < num_trials:
    position, score = game.reset()
    # Explore action space
    if random.uniform(0, 1) < epsilon:
        my_action = actions[np.random.randint(0, 4)]
        print (my_action)
        res = game.play(position, my_action)

    # Exploit learned values
    else:
        print (position)
        res = np.argmax(qtable[position]) 

    position, score = res[0], res[1]

    
    #print (board, score)







