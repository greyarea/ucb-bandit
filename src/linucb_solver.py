#!/usr/bin/env python
'''
Created on Jun 8, 2012

Linear UCB Contextual Bandit.

Based on: 
http://www.research.rutgers.edu/~lihong/pub/Li10Contextual.pdf

@author: ttuulari
'''

import math
import random
import time
import numpy as np
from scipy import linalg


T = 10000
RHO = 0.1 # Tighter upper confidence bound with smaller RHO

def alpha(rho):
    '''
    Returns alpha constant based on rho param for the UCB algorithm.
    
    Arguments: 
     - ``rho``: Rho parameter in float format
    
    >>> alpha(0.1)
    2.2238734153404085
    '''
    return 1.0 + math.sqrt(math.log(2.0/rho)/2.0)
    
def payoff(o, x, a_inv, rho):
    '''
    Calculate arm payoff based on arm parameters.
    
    Arguments: 
     - ``o``: A^-1 * b numpy array matrix
     - ``x``: Context vector x numpy array vector
     - ``a_inv``: A^-1  numpy array matrix
     - ``rho``: rho float parameter 
    '''    
    temp = np.dot(x.T, a_inv)    
    p = np.dot(o.T, x) + alpha(rho) * math.sqrt(np.dot(temp, x))
    return p[0][0]

def arm_payoff(arm, x, rho):
    '''
    Calculate payoff for arm in context, with rho.  
    
    Arguments: 
     - ``arm``: arm dict
     - ``x``: Context vector x numpy array vector
     - ``rho``: rho float parameter    
    '''    
    a_inv = arm['a_inv']    
    o = np.dot(a_inv, arm['b'])
    return payoff(o, x, a_inv, rho)

def update_a(a, context):
    '''
    Update A matrix of arm based on input context.  
    
    Arguments: 
     - ``a``: A  numpy array matrix
     - ``context``: Context vector as numpy array vector  
    '''    
    return a + np.dot(context, context.T)

def update_b(b, context, reward):
    '''
    Update b vector of arm based on input context and reward.  
    
    Arguments: 
     - ``b``: A  numpy array vector
     - ``context``: Context vector as numpy array vector  
     - ``reward``: Observer reward value as float
    '''    
    if reward != 0:
        return b + reward * context
    else:
        return b

def init(D, ARMS):
    '''    
    Initialize algorithm with input dimension D, arms ARMS.    

    Arguments: 
     - ``D``: Input dimensionality as int
     - ``ARMS``: List of arm names       
    '''        
    init_a = np.diag(np.ones(D),0)
    init_b = np.zeros((D,1))
    a_inv = linalg.inv(init_a)
    return [ {'name': a, 'a' : init_a, 
              'a_inv': a_inv, 
              'b' : init_b, 
              'total_reward' : 0.0
              } for a in ARMS ]

def payoffs(x, arms):
    '''
    Return arm payoffs in context x
    
    Arguments: 
     - ``x``: Input context as numpy array vector
     - ``arms``: List of arm dicts    
    '''
    return [(arm_payoff(arm, x, RHO), i) for i, arm in enumerate(arms)]

def best_arm(x, arms):
    '''
    Return best arm index.
    
    Arguments: 
     - ``x``: Input context as numpy array vector
     - ``arms``: List of arm dicts    
    '''    
    poffs = payoffs(x, arms)
    s = sorted(poffs, reverse = True)
    return s[0][1]        

def observe_and_update(context, reward, arms, best_arm):
    '''
    Observe context, take action based on best arm and observe reward. 
    Update arm parameters.
    
    Arguments: 
     - ``context``: Input context as numpy array vector
     - ``reward``: Float valued reward
     - ``arms``: List of arm dicts
     - ``best_arm``: Best arm index as int
    '''        
    
    a = arms[best_arm]['a']
    b = arms[best_arm]['b']
                       
    arms[best_arm]['a'] = update_a(a, context)
    arms[best_arm]['b'] = update_b(b, context, reward)
    arms[best_arm]['a_inv'] = linalg.inv(arms[best_arm]['a'])
    arms[best_arm]['total_reward'] += reward             
    
def total_reward(arms):
    '''
    Float valued total reward accumulated so far.
    
    Arguments: 
     - ``arms``: List of arm dicts    
     '''
    return sum(map(lambda d: d['total_reward'], arms))

def update_running_average(avg, context, t):
    '''
    Return updated running average.
    
    Arguments: 
    - ``avg``: Numpy array vector
    - ``context``: Input context as numpy array vector
    - ``t``: Time step t as integer
    '''     
    return (t*avg + context)/(t+1)
        
def mapper(val):
    if val < 0.5:
        return 0.0
    else:
        return 1.0

def test_rewards(t):
    rewards = np.random.random((t,1))
    return np.array([map(mapper, rewards)]).T    
    
def test2():
    D = 5
    t = 10
    rewards = test_rewards(t)
    design = np.random.random((t,D)) - 0.5
    design = np.append(np.array(np.ones((t,1))), design, axis = 1)
    ARMS = ['arm1', 'arm2']    
    (_,d) = design.shape
    arms = init(d, ARMS)
    for i in range(t):
        context = np.array([design[i,:]]).T
        best = best_arm(context, arms)
        observe_and_update(context, rewards[i][0], arms, best)
    
def test3():
    D = 2
    t = 100
    ARMS = ['arm1', 'arm2']
    arms = init(D, ARMS)
    
    for _ in range(t):
        c = random.random()    
        if c < 0.5:
            correct = 0
        elif c > 0.5:
            correct = 1
            
        context = np.array([[1.0, c]]).T
        best = best_arm(context, arms)        
        
        if best == correct:
            reward = 1.0
        else:
            reward = 0.0
        
        observe_and_update(context, reward, arms, best)
        
    print "arm1: "
    print arms[0]['total_reward']
    print len(arms[0]['contexts'])
    #print arms[0]['contexts']
    
    print "arm2: "
    print arms[1]['total_reward']
    print len(arms[1]['contexts'])
    #print arms[1]['contexts']
    
    print "total reward: " + str(arms[0]['total_reward'] + arms[1]['total_reward'])
    print "------------------------------------------------------"

def test4():
    D = 2
    ARMS = ['arm1', 'arm2', 'arm3', 'arm4']
    arms = init(D, ARMS)
    ti = time.time()
    
    avg = np.zeros(D)        
    
    for t in range(T):
        c = np.random.random(2)    
        if c[0] < 0.5 and c[1] < 0.5: 
            correct = 0
        elif c[0] < 0.5 and c[1] > 0.5:
            correct = 1
        elif c[0] > 0.5 and c[1] < 0.5:
            correct = 2
        elif c[0] > 0.5 and c[1] > 0.5:
            correct = 3
                            
        avg = update_running_average(avg, c, t)        
            
        context = np.array([[c[0] - avg[0], c[1] - avg[1]]]).T
        best = best_arm(context, arms)        
        
        if best == correct:            
            reward = 1.0
        else:
            reward = 0.0
        
        observe_and_update(context, reward, arms, best)
        
    ti = time.time() - ti 
    print "Time elapsed: " + str(ti)
    print "Time elapsed per context: " + str(ti/T)
    print "Contexts/sec: " + str(T/ti)
    
    print "arm1: "
    print arms[0]['total_reward']    
    
    print "arm2: "
    print arms[1]['total_reward']
    
    print "arm3: "
    print arms[2]['total_reward']
    
    print "arm4: "
    print arms[3]['total_reward']
    
    total_reward
    print "total reward: " + str(total_reward(arms))
    print "reward per context: " + str(total_reward(arms)/T)
    print "------------------------------------------------------"
    
def test5():
    D=3      
    T=100
    ARMS = ['health', 'exp', 'ammo']
    arms = init(D, ARMS)
    avg = np.zeros(D)  
     
    for t in range(T):
        c = random.randint(0,2)
        if c == 0:
            level = random.randint(5,10)
            health = random.randint(1,4)
            ammo = random.randint(5,10)
        elif c == 1:
            level = random.randint(1,4)
            health = random.randint(5,10)
            ammo = random.randint(5,10)            
        elif c == 2:
            level = random.randint(5,10)
            health = random.randint(5,10)
            ammo = random.randint(1,4)                                            
        
        c = np.array([level, health, ammo])
        avg = update_running_average(avg, c, t)            
        
        context = np.array([[level, health, ammo]]).T
        
        print "Store has " + ",".join(ARMS)
        print "Your exp is: " + str(level) 
        print "Your health is: " + str(health) 
        print "Your ammo is: " + str(ammo) 
    
        best = best_arm(context, arms) 
        stri = "Wanna buy " + str(ARMS[best]) + "? (y/n)"
        var = raw_input(stri)
        if var == 'y':
            reward = 1.0
        else: 
            reward = -1.0
        
        observe_and_update(context, reward, arms, best)                      

if __name__ == '__main__':
    #test()    
    #test2()
    #test3()    
    test4()
    test5()
    
