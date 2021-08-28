# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 06:20:30 2021

@author: C64990
"""

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time


seed = 100
random.seed(seed)

class mygame:
    def __init__(
            self, 
            board_size=4,
            val_max=10,
            ):
        
        self.board_size = board_size
        self.val_max = val_max,
        self.reset()
    
    def reset(self):
        self.board = np.zeros(self.board_size)
        self.score = 0
        return self.board
    
    def play(self, cell):
        board = self.board
        score = self.score
        val_max = self.val_max

        # fill in the board
        if cell >= 0 and cell < 4:
            if board[cell] < val_max:
                board[cell] = board[cell] + 1
                # evaluate the board after fill
                if board[cell] % 2 == 0:
                    score = score + 1
                else:
                    score = score - 1
            else: 
                pass

        self.board = board
        self.score = score
        return self.board, self.score


########################
# define game
board_size, val_max, = 4, 10
n, num_trials = 0, 100000
epsilon, gamma = 0.2, 1
########################


########################
# qtable creation

import itertools
listed = []
a = [0,1,2,3,4,5,6,7,8,9,10,
     0,1,2,3,4,5,6,7,8,9,10,
     0,1,2,3,4,5,6,7,8,9,10,
     0,1,2,3,4,5,6,7,8,9,10,]
for i in range(board_size, board_size+1):
   listed.append(list(itertools.permutations(a,i)))

sel = []
for elem in listed[0]:
    if elem[0] + elem[1] + elem[2] + elem[3] <= 10:
        sel.append(elem)
    else:
        pass
    
sel2 = []
for elem in sel:
    if elem in sel2:
        pass
    else:
        sel2.append(elem)
sel = sel2

sel2 = []
for elem in sel:
    if elem[0] + elem[1] + elem[2] + elem[3] == 1:
        sel2.append(elem)
    else:
        pass

rtable = pd.DataFrame(index=sel, columns=sel2)
qtable = pd.DataFrame(index=sel, columns=sel2)
for i in range(rtable.shape[0]):
    stateb = rtable.index[i]
    scoreb = sum([a % 2 for a in stateb])
    for j in range(rtable.shape[1]):
        statea = rtable.columns[j]
        statea = [sum(x) for x in zip(statea, stateb)]
        scorea = sum([a % 2 for a in statea])
        rtable.iloc[i, j] = scorea
########################



########################
# run game
game = mygame(board_size, val_max)
while n < num_trials:
    state = mygame.reset()

    # Explore action space
    if random.uniform(0, 1) < epsilon:
        passa = np.random.randint(0, 4)
        res = game.play(passa)

    # Exploit learned values
    else:
        res = np.argmax(qtable[state]) 

    next_state, reward, done, info = env.step(action) 
    board = res[0]
    score = res[1]
    q[state]
    
    print (board, score)


    
    
    passa = np.random.randint(0, 10)
    res = game.play(passa)
    board = res[0]
    score = res[1]
    print (board, score)
    n+=1





#
#"""Training the agent"""
#
#import random
#from IPython.display import clear_output
#
## Hyperparameters
#alpha = 0.1
#gamma = 0.6
#epsilon = 0.1
#
## For plotting metrics
#all_epochs = []
#all_penalties = []
#
#for i in range(1, 100001):
#    state = env.reset()
#
#    epochs, penalties, reward, = 0, 0, 0
#    done = False
#    
#    while not done:
#        if random.uniform(0, 1) < epsilon:
#            action = env.action_space.sample() # Explore action space
#        else:
#            action = np.argmax(q_table[state]) # Exploit learned values
#
#        next_state, reward, done, info = env.step(action) 
#        
#        old_value = q_table[state, action]
#        next_max = np.max(q_table[next_state])
#        
#        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
#        q_table[state, action] = new_value
#
#        if reward == -10:
#            penalties += 1
#
#        state = next_state
#        epochs += 1
#        
#    if i % 100 == 0:
#        clear_output(wait=True)
#        print(f"Episode: {i}")
#
#print("Training finished.\n")
