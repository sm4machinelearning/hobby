# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 13:49:33 2021

@author: C64990
"""

class mygame:
    def __init__(self, mat,):
        self.mat = mat
        
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
        
        adiff = mystate - astate
        bdiff = mystate - bstate
        cdiff = mystate - cstate
        
        if adiff < bdiff and adiff < cdiff:
            winner = 'a'
        if bdiff < adiff and bdiff < cdiff:
            winner = 'b'
        if cdiff < adiff and cdiff < bdiff:
            winner = 'c'
            
        if winner == 'b':
            scorenow = 10
        else:
            scorenow = -10
        return winner, scorenow


    def play(self):
        mat = self.mat
        anum = self.picknum()
        astate = self.determine_state(anum)
        bnum = self.picknum()
        bstate = self.determine_state(bnum)
        if bstate != astate:
            cnum = self.picknum()
            cstate = self.determine_state(cnum)
            if (cstate != astate) and (cstate != bstate):
                mynum = self.picknum()
                mystate = self.determine_state(mynum)
                winner, scorenow = self.calwinner(astate, bstate, cstate, mystate)
                
                if winner in ['a','b','c']:
                    mat = mat.append({'a':astate,
                                      'b':bstate,
                                      'c':cstate,
                                      'my':mystate,
                                      'winner':winner}, ignore_index=True)
    
                else:
                    pass
                
        self.mat = mat
        return self.mat


cols = ['a','b','c','my','winner']
mat = pd.DataFrame(columns=cols)
alpha, epsilon, gamma = 0.1, 0.6, 0.1

game = mygame(mat)
ntrial, numtrial = 0, 100
while ntrial < numtrial:
    mat = game.play()
    ntrial +=1
    
    
mat = mat.loc[mat['winner'] == 'b']
print (mat)
