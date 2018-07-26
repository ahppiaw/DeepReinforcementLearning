import numpy as np
import tensorflow as tf
import gym
import random

class DeepQNetwork(object):
    def __init__(self,n_actions,n_features,learning_rate=0.01,reward_decay=0.9,epsilon_greedy=0.9,epsilon_increment=0.001,replace_target_iter=300,buffer_size=500,batch_size=32):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.buffer_size = buffer_size
        self.buffer_counter = 0
        self.batch_size = batch_size
        self.epsilon = 0 if epsilon_increment is not None else epsilon_greedy
        self.epsilon_max = epsilon_greedy
        self.epsilon_increment = epsilon_increment
        self.learn_step_counter = 0
        self.buffer = np.zeros((self.buffer_size,self.n_features*2+2))
        self.bulid_net()
        target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'target_net')
        eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope = 'eval_net')
        with tf.variable_scope("soft_replacement"):
            self.target_replace_op = [tf.assign(t,e) for t,e in zip(target_params,eval_params)]
        self.sess = tf.Session()
        tf.summary.FileWriter('log/',self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
    def bulid_net(self):
        tf.reset_default_graph()
        self.s = tf.placeholder(tf.float32,[None,self.n_features])
        self._s = tf.placeholder(tf.float32,[None,self.n_features])
        self.r = tf.placeholder(tf.float32,[None,])
        self.a = tf.placeholder(tf.int32,[None,])
        w_init = tf.random_normal_initializer(0.,0.3)
        b_init = tf.constant_initializer(0.1)
        with tf.variable_scope("eval_net"):
            eval_layer = tf.layers.dense(self.s,20,tf.nn.relu,kernel_initializer=w_init,bias_initializer=b_init,name='eval_layer')
            self.q_eval = tf.layers.dense(eval_layer,self.n_actions,kernel_initializer=w_init,bias_initializer=b_init,name='output_layer1')
        with tf.variable_scope("target_net"):
            target_layer = tf.layers.dense(self._s,20,tf.nn.relu,kernel_initializer=w_init,bias_initializer=b_init,name='target_layer')
            self.q_next = tf.layers.dense(target_layer,self.n_actions,kernel_initializer=w_init,bias_initializer=b_init,name='output_layer2')
        with tf.variable_scope("q_target"):
            self.q_target = tf.stop_gradient(self.r + self.gamma*tf.reduce_max(self.q_next,axis=1))
        with tf.variable_scope("q_eval"):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0]),self.a],axis=1)
            self.q_eval_a = tf.gather_nd(params=self.q_eval,indices=a_indices)
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval_a))
        with tf.variable_scope("train"):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
    def store_transition(self,s,a,r,_s):
        transition = np.hstack((s,a,r,_s))
        index = self.buffer_counter % self.buffer_size
        self.buffer[index,:] = transition
        self.buffer_counter += 1
    def egreedy_action(self,state):
        state = state[np.newaxis, :]
        if random.random() <self.epsilon:
            action_value = self.sess.run(self.q_eval,feed_dict={self.s:state})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0,self.n_actions)
        return action
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter ==0:
            self.sess.run(self.target_replace_op)
        sample_index = np.random.choice(min(self.buffer_counter,self.buffer_size),size = self.batch_size)
        batch_buffer = self.buffer[sample_index,:]
        _, cost = self.sess.run([self.train_op,self.loss],feed_dict={self.s : batch_buffer[:,:self.n_features],self.a : batch_buffer[:, self.n_features],self.r : batch_buffer[:,self.n_features+1],self._s : batch_buffer[:,-self.n_features:]})
        self.epsilon = min(self.epsilon_max,self.epsilon+self.epsilon_increment)
        self.learn_step_counter+=1
        return cost

max_episode = 100
env = gym.make("CartPole-v0")
env = env.unwrapped
import time
RL = DeepQNetwork(n_actions=env.action_space.n,n_features=env.observation_space.shape[0])
total_step = 0
for episode in range(max_episode):
    state = env.reset()
    episode_reward = 0
    while True:
        if episode%10 == 0:
            env.render()
            time.sleep(0.01)
        act = RL.egreedy_action(state)
        observation_,reward,done,info = env.step(act)
        x,x_dot,theta,theta_dot = observation_
        reward1 = (env.x_threshold-abs(x))/env.x_threshold - 0.8
        reward2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = reward1 + reward2
        RL.store_transition(state,act,reward,observation_)
        if total_step>100:
           cost = RL.learn()
           print("cost is %.3f"%cost)
        episode_reward += reward
        state = observation_
        if done:
            print("episode:",episode)
            print("episode_reward:%.3f"%episode_reward)
            break
        total_step += 1
        
