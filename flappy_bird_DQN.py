#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 09:35:27 2018
about bird
@author: ahppiaw
"""

import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
from BrainDQN_Nature import BrainDQN
import numpy as np

def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation,(80,80)),cv2.COLOR_BGR2GRAY)
    ret,observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,(80,80,1))

def playlittlebird():
    actions = 2
    agent = BrainDQN(actions)
    flappy_bird = game.GameState()
    
    action0 = np.array([1,0])
    observation0,reward0,terminal = flappy_bird.frame_step(action0)
    observation0 = cv2.cvtColor(cv2.resize(observation0,(80,80)),cv2.COLOR_BGR2GRAY)
    ret,observation0 = cv2.threshold(observation0,1,255,cv2.THRESH_BINARY)
    agent.Initstate(observation0)
    
    while 1 != 0:
        action = agent.greedy_action()
        nextobservation,reward,terminal = flappy_bird.frame_step(action)
        nextobservation = preprocess(nextobservation)
        agent.store(nextobservation,action,reward,terminal)
        
def main():
    playlittlebird()
    
if __name__ == "__main__":
    main()