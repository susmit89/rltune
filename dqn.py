# -*- coding: utf-8 -*-
import random
import gym
import gym_dbenv
import numpy as np
import pickle
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('GTK')
EPISODES = 500

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.999
        self.learning_rate = 0.1
        self.model = self._build_model()
        #self.target_model = self._build_model()
        #self.update_target_model()

    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate), metrics=['mae'])
        return model

    #def update_target_model(self):
        # copy weights from model to target_model
    #    self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        print("model predicted action")
        return np.argmax(act_values[0])  # returns action

    def net_update(self, state, action, reward, next_state, done):
        t = self.model.predict(next_state)
        target = reward + self.gamma * np.amax(t)
        print target
        target_vec = self.model.predict(state)
        target_vec[0][action] = target
        print target_vec
        self.model.fit(state, target_vec, epochs=1, verbose=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        i=0
        for state, action, reward, next_state, done in minibatch:
            print(i)
            i = i+1
            target = self.model.predict(state)
            #if done:
            #    target[0][action] = reward
            #else:
                # a = self.model.predict(next_state)[0]
            #t = self.model.predict(next_state)[0]
            #print t
            #print(len(t))
            target[0][action] = reward + self.gamma * np.amax(target)
            print target
            print(len(target[0]))
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state, target, epochs=1, verbose=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)
        with open ('qfile', 'rb') as fp:
             self.memory = pickle.load(fp)

    def save(self, name):
        self.model.save_weights(name)
        with open('qfile', 'wb') as fp:
             pickle.dump(self.memory, fp)





if __name__ == "__main__":


    env = gym.make('DB-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    ## mathplot
    plot_1 = plt.subplot(211)
    plt.xlabel('Set Indexed')
    plt.ylabel('Steps')
    plt.xticks(range(env.col_len),
       list(zip(*env.t_columns)[1]), fontsize=4)
    plot_2 = plt.subplot(212)
    plt.ylabel('exploration rate')
    plt.xlabel('Steps')
    #plt.yticks(range(env.col_len),
    #   list(env.t_columns))

    try:
      agent.load("db_model.h5")
    except:
      pass
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        t_reward = 0
        for time in range(500):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            t_reward = t_reward + reward
            #reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            agent.net_update(state, action, reward, next_state, done)
            state = next_state
            plot_1.scatter(action,e,10)
            plot_2.scatter(e,agent.epsilon,10)
            plt.pause(0.05)
            #if len(agent.memory) > batch_size:
            #    agent.replay(batch_size)
            if done:
                agent.remember(state, action, reward, next_state, done)
                #agent.update_target_model()
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break



        print "Total REWARD =",  t_reward
        if e % 10 == 0:
             plt.savefig("figure1.ps")
             agent.save("db_model.h5")
