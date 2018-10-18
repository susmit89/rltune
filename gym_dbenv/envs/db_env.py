import gym
from gym import error, spaces, utils
from client import DBENGINE
import numpy as np
import random
import re

class DBENV(gym.Env):

    def __init__(self):
        self.db = DBENGINE()
        self.queries = np.array(self.db.get_query_workload().splitlines())
        print(self.queries)
        self.t_columns = self.db.column_schema()
        self.n_columns = self.db.get_column_count()
        self.t_tables=tuple([i[1]+1 for i in self.n_columns])
        low = np.zeros(len(self.t_tables), dtype=int)
        high =  np.array(self.t_tables, dtype=int) - np.ones(len(self.t_tables), dtype=int)
        self.action_space = spaces.Discrete(len(self.t_columns))
        self.observation_space = spaces.Box(-1, 1, shape=(2*len(self.t_columns),), dtype=int)
        self.state = None
        self.index_count = 4

    def step(self, action):
        #print("Action: %s",action)
        #self.db.create_index(self.t_columns[32])
        n = random.randint(0,len(self.queries)-1)
        print n
        query = self.queries[n]
        print self.queries[n]
        table = re.search(r"FROM\s(.*)WHERE", query).groups()
        print "table", table
        sub = re.search(r"WHERE\s(.*)", query).groups()[0]
        col = re.sub(" \d+|AND|>|=|<|.\d+|FOR\s(.*)", " ", sub).split()
        print "column", col
        print("count",self.index_count)
        if self.index_count == -1:
           done = True
        else:
           done = False
        self.index_count = self.index_count - 1
        reward = 1
        self.state = np.zeros(2*len(self.t_columns), dtype=int)
        info = {}
        return self.state, reward, done, info

    def reset(self):
        self.index_count = 4
        self.db.clear_index()
        np.random.shuffle(self.queries)
        self.done = False
        return np.zeros(2*len(self.t_columns), dtype=int)

    def render(self, mode='human'):
        print("Rendering")
        return 1
