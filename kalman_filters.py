# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 06:02:45 2021

@author: C64990
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def upd_v(u, a, t):
    return u + a*t

def upd_s(u, a, t):
    return u*t + 0.5*a*t**2

def upd_x_u(x0, u0, a, tau):
    if u0 > 0:
        delx = upd_s(u0, a, tau)
        v = upd_v(u0, a, tau)
        return x0 + delx, v
    else:
        delx = upd_s(u0, a, tau)
        if x0 + delx > 0:
            v = upd_v(u0, a, tau)
            return x0 + delx, v
        else:
            deriv = np.sqrt(u0**2 - 2*a*x0)
            times = [(-u0 + deriv) / a, (-u0 - deriv) / a]
            if times[0] > 0:
                thit = times[0]
            else:
                thit = times[1]
            
            vhit = upd_v(u0, a, thit)
            vnew = -1 * vhit
            x1 = upd_s(vnew, a, tau-thit)
            v1 = upd_v(vnew, a, tau-thit)
            return x1, v1
            
x0, u0, a = 20, -5, -9.8
delt, ntime = 0.1, 100
traj = np.zeros((ntime, 4))
traj[0,0], traj[0,1], traj[0,2], traj[0,3] = 0, x0, u0, a

x = 1
while x < ntime:
    x1, u1 = upd_x_u(x0, u0, a, delt)
    traj[x,0] = delt*x
    traj[x,1] = x1
    traj[x,2] = u1
    traj[x,3] = a
    x0 = x1
    u0 = u1
    x+=1
    
#plt.plot(traj[:,0], traj[:,1])
#plt.plot(traj[:,0], traj[:,2])
#traj = traj.round(1)


"""
What is a Kalman Filter?
The Kalman filter is an algorithm that uses noisy observations of a system 
over time to estimate the parameters of the system (some of which are 
unobservable) and predict future observations. At each time step, it 
makes a prediction, takes in a measurement, and updates itself based 
on how the prediction and measurement compare.

The algorithm is as follows:
    
. Take as input a mathematical model of the system, i.e.
the transition matrix, which tells us how the system evolves from one state 
to another. For instance, if we are modeling the movement of a car, then the 
next values of position and velocity can be computed from the previous ones 
using kinematic equations. Alternatively, if we have a system which is fairly 
stable, we might model its evolution as a random walk. If you want to read up 
on Kalman filters, note that this matrix is usually called 

. the observation matrix, which tells us the next measurement we should expect 
given the predicted next state. If we are measuring the position of the car, 
we just extract the position values stored in the state. For a more complex 
example, consider estimating a linear regression model for the data. 
Then our state is the coefficients of the model, and we can predict the next 
measurement from the linear equation. This is denoted 

. any control factors that affect the state transitions but are not part of 
the measurements. For instance, if our car were falling, gravity would be a 
control factor. If the noise does not have mean 0, it should be shifted over 
and the offset put into the control factors. The control factors are 
summarized in a matrix B with time-varying control vector ut, which give the 
offset But

. covariance matrices of the transition noise (i.e. noise in the evolution of 
the system) and measurement noise, denoted Q and R, respectively. Take as input 
an initial estimate of the state of the system and the error of the estimate, 
0 μ0 and 0 σ0

. At each timestep:
estimate the current state of the system 
xt using the transition matrix take as input new measurements zt
use the conditional probability of the measurements given the state, taking 
into account the uncertainties of the measurement and the state estimate, 
to update the estimated current state of the system xt and the covariance 
matrix of the estimate Pt

This graphic illustrates the procedure followed by the algorithm. 
It's very important for the algorithm to keep track of the covariances of 
its estimates. This way, it can give us a more nuanced result than simply 
a point value when we ask for it, and it can use its confidence to decide 
how much to be influenced by new measurements during the update process. 
The more certain it is of its estimate of the state, the more skeptical 
it will be of measurements that disagree with the state.
By default, the errors are assumed to be normally distributed, and this 
assumption allows the algorithm to calculate precise confidence intervals. 
It can, however, be implemented for non-normal errors.

"""

x0, u0 = 20, -5
t0, tfin, tau = 0, 20, 1

state0 = [x0, u0]
tran_mat = [[1, tau],
            [0, 1]]

time = np.arange(t0, tfin, tau)
traj = np.zeros((2, (len(time))))
traj[:,0] = state0

x = 1
while x < tfin:
    traj[:,x] = np.matmul(tran_mat, traj[:,x-1])
    x+=1
    


'''

Example: Estimating Moving Average¶
Because the Kalman filter updates its estimates at every time step and tends 
to weigh recent observations more than older ones, it can be used to estimate 
rolling parameters of the data. When using a Kalman filter, there's no window 
length that we need to specify. This is useful for computing the moving average 
or for smoothing out estimates of other quantities.
Below, we'll use both a Kalman filter and an n-day moving average to estimate 
the rolling mean of a dataset. We construct the inputs to the Kalman filter 
as follows:
The mean is the model's guess for the mean of the distribution from which 
measurements are drawn. This means our prediction of the next value is equal 
to our estimate of the mean. 
Hopefully the mean describes our observations well, hence it shouldn't change 
significantly when we add an observation. This implies we can assume that it 
evolves as a random walk with a small error term. We set the transition matrix 
to 1 and transition covariance matrix is a small number.
We assume that the observations have variance 1 around the rolling mean (1 is 
chosen randomly). 
Our initial guess for the mean is 0, but the filter realizes that that is 
incorrect and adjusts.
'''

from pykalman import KalmanFilter
datafile = '../stock_data/AAPL.csv'
data = pd.read_csv(datafile)
data = data.dropna(axis='rows')

data.columns = [x.lower() for x in data.columns]
data = data.rename(columns={'adj close' : 'adj_close',
                            'date' : 'initdate'})
data['initdate'] = pd.to_datetime(data['initdate'])
data = data.set_index('initdate')
dataa = data[['adj_close']]

mu, sigma = 0.001, 0.0001 
noise = np.random.normal(mu, sigma, dataa.shape) 
datam = dataa + noise
x = dataa/datam

# Construct a Kalman filter
kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = 0,
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01)

# Use the observed values of the price to get a rolling mean
state_means, _ = kf.filter(x.values)
state_means = pd.Series(state_means.flatten(), index=x.index)

# Compute the rolling mean with various lookback windows
mean30 = x.rolling(window = 10).mean()
mean60 = x.rolling(window = 30).mean()
mean90 = x.rolling(window = 60).mean()

# Plot original data and estimated mean
plt.figure(figsize=(15,7))
plt.plot(state_means[60:], '-b', lw=2, )
plt.plot(x[60:],'-g',lw=1.5)
plt.plot(mean30[60:], 'm', lw=1)
plt.plot(mean60[60:], 'y', lw=1)
plt.plot(mean90[60:], 'c', lw=1)
plt.title('Kalman filter estimate of average')
plt.legend(['Kalman Estimate', 'X', '30-day Moving Average', '60-day Moving Average','90-day Moving Average'])
plt.xlabel('Day')
plt.ylabel('Price');


'''
We'll be using the z score in the same way as before. Our strategy is to go 
long or short only in the areas where the |error| is greater than one 
standard deviation. Since 1 day price could be noisy, we'll be using 5 day 
average for a particular day's price

Let's modify our trading function to make use of Kalman Filter while keeping 
the same logic for carrying out trades
'''

def trade(S1, S2):
    # Compute rolling mean and rolling standard deviation
    ratios = S1/S2

    kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = 0,
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.001)
    
    state_means, state_cov = kf.filter(ratios.values)
    state_means, state_std = state_means.squeeze(), np.std(state_cov.squeeze())
    
    window = 5
    ma = ratios.rolling(window=window, center=False).mean()
    zscore = (ma - state_means)/state_std
    
    # Simulate trading
    # Start with no money and no positions
    money = 0
    countS1 = 0
    countS2 = 0
    for i in range(len(ratios)):
        # Sell short if the z-score is > 1
        if zscore[i] > 1:
            money += S1[i] - S2[i] * ratios[i]
            countS1 -= 1
            countS2 += ratios[i]
        # Buy long if the z-score is < 1
        elif zscore[i] < -1:
            money -= S1[i] - S2[i] * ratios[i]
            countS1 += 1
            countS2 -= ratios[i]
        # Clear positions if the z-score between -.5 and .5
        elif abs(zscore[i]) < 0.5:
            money += countS1*S1[i] + S2[i] * countS2
            countS1 = 0
            countS2 = 0
#         print('Z-score: '+ str(zscore[i]), countS1, countS2, S1[i] , S2[i])
    return money

result = trade(dataa, datam)
print (result)









