#!/usr/bin/env python
'''
Created on Jun 8, 2012

Linear UCB Contextual Bandit prototype

Based on: 
http://www.research.rutgers.edu/~lihong/pub/Li10Contextual.pdf

@author: ttuulari
'''

import math
import random
import numpy as np
from scipy import linalg


T = 1000
RHO = 0.1

def alpha(rho):
    return 1.0 + math.sqrt(math.log(2.0/rho)/2.0)
    
def A(design):
    (_,d) = design.shape
    return np.dot(design.T, design) + np.diag(np.ones(d),0)

def params(a, b):
    i = linalg.inv(a)
    return (np.dot(i,b), i)

def payoff(o, x, a_inv, rho):
    temp = np.dot(x.T, a_inv)    
    p = np.dot(o.T, x) + alpha(rho) * math.sqrt(np.dot(temp, x))
    return p[0][0]

def arm_payoff(arm, x, rho):
    a_inv = arm['a_inv']    
    o = np.dot(a_inv, arm['b'])
    return payoff(o, x, a_inv, rho)

def update_a(a, context):
    return a + np.dot(context, context.T)

def update_b(b, context, reward):
    return b + reward * context

def init(D, ARMS):
    init_a = np.diag(np.ones(D),0)
    init_b = np.zeros((D,1))
    a_inv = linalg.inv(init_a)
    return [ {'name': a, 'a' : init_a, 
              'a_inv': a_inv, 
              'b' : init_b, 
              'total_reward' : 0.0,
              'contexts' : [] } for a in ARMS ]

def payoffs(x, arms):
    return [(arm_payoff(arm, x, RHO), i) for i, arm in enumerate(arms)]

def best_arm(x, arms):
    poffs = payoffs(x, arms)
    s = sorted(poffs, reverse = True)
    return s[0][1]        

def observe_and_update(context, reward, arms, best_arm):
    a = arms[best_arm]['a']
    b = arms[best_arm]['b']
                       
    arms[best_arm]['a'] = update_a(a, context)
    arms[best_arm]['b'] = update_b(b, context, reward)
    arms[best_arm]['a_inv'] = linalg.inv(arms[best_arm]['a'])
    arms[best_arm]['total_reward'] += reward
    arms[best_arm]['contexts'].append(context)               

def mapper(val):
    if val < 0.5:
        return 0.0
    else:
        return 1.0

def test_rewards():
    rewards = np.random.random((T,1))
    return np.array([map(mapper, rewards)]).T    

def iter_test(design, rewards):
    (_,d) = design.shape    
    a = np.diag(np.ones(d),0)
    b = np.zeros((d,1))    
    
    for i in range(T):
        context = np.array([design[i,:]]).T        
        a = update_a(a, context)
        b = update_b(b, context, rewards[i])
        
    print "A: " + str(a)
    print "b: " + str(b)
    
    return (a, b)

def test():
    D = 5
    rewards = test_rewards()                                                    
    design = np.random.random((T,D))
    a = A(design)
    print "A: " + str(a)
    (a_it, _) = iter_test(design, rewards)
    assert( math.fabs(sum(a.flatten(1) - a_it.flatten(1))) < 1e-10 )
    
def test2():
    D = 5
    rewards = test_rewards()
    design = np.random.random((T,D))
    ARMS = ['arm1', 'arm2']
    arms = init(D, ARMS)
    for i in range(T):
        context = np.array([design[i,:]]).T
        best = best_arm(context, arms)
        observe_and_update(context, rewards[i][0], arms, best)
    
def test3():
    D = 1
    ARMS = ['arm1', 'arm2']
    arms = init(D, ARMS)
    
    for _ in range(T):
        c = random.random()    
        if c < 0.5:
            correct = 0
        elif c > 0.5:
            correct = 1
            
        context = np.array([[c-0.5]]).T
        best = best_arm(context, arms)        
        
        if best == correct:
            reward = 1.0
        else:
            reward = 0.0
        
        observe_and_update(context, reward, arms, best)
    
    print "D: 1"
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
    D=2        
    ARMS = ['arm1', 'arm2', 'arm3', 'arm4']
    arms = init(D, ARMS)
    
    for _ in range(T):
        c = (random.random(), random.random())    
        if c[0] < 0.5 and c[1] < 0.5: 
            correct = 0
        elif c[0] < 0.5 and c[1] > 0.5:
            correct = 1
        elif c[0] > 0.5 and c[1] < 0.5:
            correct = 2
        elif c[0] > 0.5 and c[1] > 0.5:
            correct = 3
            
        context = np.array([[c[0]-0.5, c[1]-0.5]]).T
        best = best_arm(context, arms)        
        
        if best == correct:
            reward = 1.0
        else:
            reward = 0.0
        
        observe_and_update(context, reward, arms, best)
        
    print "D: 2"
    print "arm1: "
    print arms[0]['total_reward']
    print len(arms[0]['contexts'])
    #print arms[0]['contexts']
    
    print "arm2: "
    print arms[1]['total_reward']
    print len(arms[1]['contexts'])
    #print arms[1]['contexts']
    
    print "arm3: "
    print arms[2]['total_reward']
    print len(arms[2]['contexts'])
    #print arms[2]['contexts']
    
    print "arm4: "
    print arms[3]['total_reward']
    print len(arms[3]['contexts'])
    #print arms[3]['contexts']
    
    print "total reward: " + str(arms[0]['total_reward'] + arms[1]['total_reward'] + arms[2]['total_reward'] + arms[3]['total_reward'])
    print "------------------------------------------------------"
    

if __name__ == '__main__':
    test()    
    test2()
    test3()
    test4()
