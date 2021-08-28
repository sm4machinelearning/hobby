# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:06:25 2021

@author: C64990
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations  
np.random.seed(10)


class mygame:
    def __init__(
            self, 
            mat,
            positionsdict
            ):
        self.mat = mat
        self.positiondict = positionsdict
        self.reset()
    
    def reset(self):
        cols = ['anum','bnum','cnum','mynum','winner']
        mat = pd.DataFrame(columns=cols)
        self.score = 0
        return self.mat, self.score
    
    def picknum(self):
        return np.random.uniform(0, 1, 1)[0]
    
    def calscore(self, winner):
        if winner == 'b':
            scorenow = 10
        else:
            scorenow = -10
        return self.score + scorenow
    
    def getstate(self,a,b,c,my,winner):
        posdict = self.positiondict
        d = {'a': a, 'b': b, 'c': c, 'my':my}
        d = [key for (key, value) in sorted(d.items(), key=lambda key_value: key_value[1])]
        d = tuple(d)
        for key, value in posdict.items():
            if d == value:
                ans = key
        if winner == 'a':
            return ans + 'a'
        if winner == 'b':
            return ans + 'b'
        if winner == 'c':
            return ans + 'c'        
    
    def play(self):
        mat = self.mat
        anum = self.picknum()
        bnum = self.picknum()
        if bnum != anum:
            cnum = self.picknum()
            if (cnum != anum) and (cnum != bnum):
                mynum = self.picknum()
                adiff = abs(mynum - anum)
                bdiff = abs(mynum - bnum)
                cdiff = abs(mynum - cnum)
                
                if adiff < bdiff and adiff < cdiff:
                    winner = 'a'
                if bdiff < adiff and bdiff < cdiff:
                    winner = 'b'
                if cdiff < adiff and cdiff < bdiff:
                    winner = 'c'
                
                state = self.getstate(anum, bnum, cnum, mynum, winner)
                
                mat = mat.append({'anum':anum,
                                  'bnum':bnum,
                                  'cnum':cnum,
                                  'mynum':mynum,
                                  'winner':winner}, ignore_index=True)
                
        self.mat = mat
        self.score = self.calscore(winner)
        return self.mat, self.score, state
    
    
cols = ['anum','bnum','cnum','mynum','winner']
mat = pd.DataFrame(columns=cols)
alpha, epsilon, gamma = 0.1, 0.6, 0.1

arr = ['a','b','c','my']
allp = list(permutations(arr))
allps = {}
psforqt = []
for p in range(len(allp)):
    allps['s' + str(p)] = allp[p]
    psforqt.append('s' + str(p) + 'a')
    psforqt.append('s' + str(p) + 'b')
    psforqt.append('s' + str(p) + 'c')

qtable = pd.DataFrame(index=psforqt, columns=['action'])

game = mygame(mat, allps)
ntrial, numtrial = 0, 10000
while ntrial < numtrial:
    if np.random.uniform(0, 1, 1)[0] < epsilon:
        mat, score, state = game.play()
    else:
        mat, score, state = np.argmax(qtable[state])
    
    

    ntrial +=1
print (mat['winner'].value_counts())
    
mat = mat.loc[mat['winner'] == 'b']
plt.hist(mat['anum'], 50)
plt.hist(mat['bnum'], 50)
plt.hist(mat['cnum'], 50)



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

