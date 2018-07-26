#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 14:08:55 2018
"Playing Atari with Deep Reinforcement Learning"
@author: ahppiaw
"""

import tensorflow as tf
import numpy as np
import random
from collections import deque

# Hyper Parameters
FRAME_PRE_ACTION = 1
GAMMA = 0.9
OBSERVE = 100.#50000.
EXPLORE = 200000.#1000000.
FINAL_EPSILON = 0.001#0.1
INITIAL_EPSILON = 0.01#1.0
REPLAY_MEMORY = 50000#1000000
BATCH_SIZE = 32
UPDATE_TIME = 100#10000
#LEARNING_RATE = 0.00025

class BrainDQN(object):
    
    def __init__(self,actions):
        self.replay_memory = deque()
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.actions = actions
        
        self.state_input,self.q_value,self.w_conv1,self.b_conv1,self.w_conv2,self.b_conv2,self.w_conv3,self.b_conv3,self.w_fc1,self.b_fc1,self.w_fc2,self.b_fc2 = self.createQnetwork()
        self.tstate_input,self.tq_value,self.tw_conv1,self.tb_conv1,self.tw_conv2,self.tb_conv2,self.tw_conv3,self.tb_conv3,self.tw_fc1,self.tb_fc1,self.tw_fc2,self.tb_fc2 = self.createQnetwork()
        self.replace_target = [self.tw_conv1.assign(self.w_conv1),self.tb_conv1.assign(self.b_conv1),self.tw_conv2.assign(self.w_conv2),self.tb_conv2.assign(self.b_conv2),self.tw_conv3.assign(self.w_conv3),self.tw_fc1.assign(self.w_fc1),self.tb_fc1.assign(self.b_fc1),self.tw_fc2.assign(self.w_fc2),self.tb_fc2.assign(self.b_fc2)]
        
        self.create_train_method()
        
        self.saver = tf.train.Saver()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state("save_network")
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess,ckpt.model_checkpoint_path)
            print("successfully loaded:")
        else:
            print("could not find old network weight")
    
    def createQnetwork(self):
        w_conv1 = self.weight_variable([8,8,4,32])
        b_conv1 = self.bias_variable([32])
        w_conv2 = self.weight_variable([4,4,32,64])
        b_conv2 = self.bias_variable([64])
        w_conv3 = self.weight_variable([3,3,64,64])
        b_conv3 = self.bias_variable([64])
        w_fc1 = self.weight_variable([1600,512])
        b_fc1 = self.bias_variable([512])
        w_fc2 = self.weight_variable([512,self.actions])
        b_fc2 = self.bias_variable([self.actions])
        
        state_input = tf.placeholder(tf.float32,[None,80,80,4])
        
        h_conv1 = tf.nn.relu(self.conv2d(state_input,w_conv1,4)+b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1,w_conv2,2)+b_conv2)
        h_conv3 = tf.nn.relu(self.conv2d(h_conv2,w_conv3,1)+b_conv3)
        h_conv3_flat = tf.reshape(h_conv3,[-1,1600])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,w_fc1)+b_fc1)
        q_value = tf.matmul(h_fc1,w_fc2)+b_fc2
        
        return state_input,q_value,w_conv1,b_conv1,w_conv2,b_conv2,w_conv3,b_conv3,w_fc1,b_fc1,w_fc2,b_fc2
    
    def replace(self):
        self.sess.run(self.replace_target)
    
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape,stddev = 0.01)
        return tf.Variable(initial)
    
    def bias_variable(self,shape):
        initial = tf.constant(0.01,shape = shape)
        return tf.Variable(initial)
    
    def conv2d(self,x,w,stride):
        return tf.nn.conv2d(x,w,strides = [1,stride,stride,1],padding = "SAME")

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = "SAME")
    
    def create_train_method(self):
        self.action_input = tf.placeholder(tf.float32,[None,self.actions])
        self.yi_input = tf.placeholder(tf.float32,[None])
        q_action = tf.reduce_sum(tf.multiply(self.q_value,self.action_input),reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.squared_difference(self.yi_input,q_action))
        self.train_step = tf.train.RMSPropOptimizer(1e-6).minimize(self.cost)
    
    def trainQnetwork(self):
        minibatch = random.sample(self.replay_memory,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        nextstate_batch = [data[3] for data in minibatch]
        
        yi_batch = []
        q_value_batch = self.sess.run(self.tq_value,feed_dict={self.tstate_input:nextstate_batch})
        for i in range(BATCH_SIZE):
            terminal = minibatch[i][4]
            if terminal:
                yi_batch.append(reward_batch[i])
            else :
                yi_batch.append(reward_batch[i] + GAMMA*np.max(q_value_batch[i]))
        self.sess.run(self.train_step,feed_dict={self.action_input:action_batch,
                                                 self.state_input:state_batch,
                                                 self.yi_input:yi_batch})
        if self.time_step % 10000 == 0:
            self.saver.save(self.sess,"save_network/"+"network"+"-dqn",global_step = self.time_step)
        if self.time_step % UPDATE_TIME == 0:
            self.replace()
    
    def Initstate(self,observation):
        self.currentstate = np.stack((observation,observation,observation,observation),axis = 2)
    
    def store(self,nextobservation,action,reward,terminal):
        new_state = np.append(nextobservation,self.currentstate[:,:,:-1],axis = 2)
        self.replay_memory.append((self.currentstate,action,reward,new_state,terminal))
        if len(self.replay_memory) >REPLAY_MEMORY:
            self.replay_memory.popleft()
        if self.time_step > OBSERVE :
            self.trainQnetwork()
        self.currentstate = new_state
        self.time_step += 1
    
    def greedy_action(self):
        qvalue = self.q_value.eval(feed_dict={self.state_input:[self.currentstate]})[0]
        action = np.zeros(self.actions)
        a_index = 0
        if self.time_step % FRAME_PRE_ACTION == 0:
            if random.random() <= self.epsilon :
                a_index = random.randrange(self.actions)
                action[a_index] = 1
            else:
                a_index = np.argmax(qvalue)
                action[a_index] = 1
        else:
            action[0] = 1
        if self.epsilon > FINAL_EPSILON and self.time_step >OBSERVE:
            self.epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EXPLORE
        return action
    
