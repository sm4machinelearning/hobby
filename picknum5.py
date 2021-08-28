# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 13:49:33 2021

@author: C64990
"""

import numpy as np
import pandas as pd

np.random.seed(100)
class mygame:
#    def __init__(self):

    def picknum(self):
        return np.random.uniform(0, 1, 1)[0]

    def determine_state(self, val):
        if val == 0:
            state = 's1'
        elif val > 0 and val < 0.1:
            state = 's2'
        elif val == 0.1:
            state = 's3'
        elif val > 0.1 and val < 0.2:
            state = 's4'
        elif val == 0.2:
            state = 's5'
        elif val > 0.2 and val < 0.3:
            state = 's6'
        elif val == 0.3:
            state = 's7'
        elif val > 0.3 and val < 0.4:
            state = 's8'
        elif val == 0.4:
            state = 's9'
        elif val > 0.4 and val < 0.5:
            state = 's10'
        elif val == 0.5:
            state = 's11'
        elif val > 0.5 and val < 0.6:
            state = 's12'
        elif val == 0.6:
            state = 's13'
        elif val > 0.6 and val < 0.7:
            state = 's14'
        elif val == 0.7:
            state = 's15'
        elif val > 0.7 and val < 0.8:
            state = 's16'
        elif val == 0.8:
            state = 's17'
        elif val > 0.8 and val < 0.9:
            state = 's18'
        elif val == 0.9:
            state = 's19'
        elif val > 0.9 and val < 1.0:
            state = 's20'
        elif val == 1.0:
            state = 's21'
        return state

    def calwinner(self, astate, bstate, cstate, mystate):
        astate = int(astate[1:])
        bstate = int(bstate[1:])
        cstate = int(cstate[1:])
        mystate = int(mystate[1:])
        
        adiff = abs(mystate - astate)
        bdiff = abs(mystate - bstate)
        cdiff = abs(mystate - cstate)
        
        if adiff < bdiff and adiff < cdiff:
            winner = 'a'
            scorenow = -10
            return winner, scorenow
            
        if bdiff < adiff and bdiff < cdiff:
            winner = 'b'
            scorenow = 10
            return winner, scorenow
            
        if cdiff < adiff and cdiff < bdiff:
            winner = 'c'
            scorenow = -10
            return winner, scorenow            
        
        else:
            return 'nowinner', 'noreward'


    def gen_state(self):
        
        state_generated = False
        while state_generated == False:    
            anum = self.picknum()
            anum = 0.0
            astate = self.determine_state(anum)
    
            bnum = self.picknum()
            bstate = self.determine_state(bnum)
    
            if bstate != astate:
                cnum = self.picknum()
                cstate = self.determine_state(cnum)
    
                if (cstate != astate) and (cstate != bstate):
                    statenow = '_'.join([astate, bstate, cstate])
                    state_generated = True
        return statenow
    
    def play(self, state):
        statenow = state
        mynum = self.picknum()
        mystate = self.determine_state(mynum)

        return mystate
    
    def step(self, state, action):
        astate, bstate, cstate = state.split('_')
        winner, reward = self.calwinner(astate, bstate, cstate, action)
        return reward
    
arr = np.arange(1, 22)
arr = ['s' + str(s) for s in arr]
from itertools import permutations
allcombinations = list(permutations(arr, 3))
allcombinations = [list(x) for x in allcombinations]
allstates = ['_'.join(x) for x in allcombinations]
qtable = pd.DataFrame(index=allstates, columns=arr)
qtable = qtable.fillna(0.)


# Hyperparameters
epsilon = 0.2 # when to learn
alpha = 0.2 # learning rate
gamma = 0.5 # importance of reward

game = mygame()
ntrial, numtrial = 0, 1000000


state = game.gen_state()
while ntrial < numtrial:
    
    if np.random.uniform(0, 1, 1)[0] < epsilon:
        action = game.play(state)
    else:
        action = np.argmax(qtable.loc[state])
        
    reward = game.step(state, action)
    
    if reward != 'noreward':
        old_value = qtable.loc[state, action]
    
        next_state = game.gen_state()
        next_max = np.max(qtable.loc[next_state])        
    
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        qtable.loc[state, action] = new_value
        state = next_state
        ntrial +=1
    
    else:
        ntrial +=1
        pass

qtable['sum'] = qtable[arr].sum(axis=1,skipna=True)
qtable['avg'] = qtable[arr].mean(axis=1,skipna=True)
qtable['max'] = qtable[arr].max(axis=1,skipna=True)
qtable = qtable.sort_values(by=['max','sum','avg'], ascending=False)

qtable.to_csv(r'C:\Users\C64990\hobby\bwin_azero.csv')
#qtablesum.to_csv(r'C:\Users\C64990\hobby\awin.csv')




"""
Evaluate the model
"""

total_epochs, total_penalties = 0, 0
episodes = 100

epochs, penalties, reward = 0, 0, 0

while epochs < episodes:
    action = np.argmax(qtable.loc[state])
    reward = game.step(state, action)

    if reward != 'noreward':
        if reward == -10:
            penalties += 1
        epochs += 1

total_penalties = penalties
total_epochs = epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")




'''
%%time
"""Training the agent"""

import random
from IPython.display import clear_output

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False
    
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action) 
        
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1
        
    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")

print("Training finished.\n")
'''


