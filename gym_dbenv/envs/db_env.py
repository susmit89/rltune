import gym
from gym import error, spaces, utils
from client import DBENGINE
import numpy as np
import random
import re
import json

INDEX_LIMIT = 10

class DBENV(gym.Env):

    def __init__(self):
        self.db = DBENGINE()
        self.queries = np.array(self.db.get_query_workload().splitlines())

        self.t_columns = self.db.column_schema()
        self.col_len = len(self.t_columns)
        self.query_len = len(self.queries)
        self.n_columns = self.db.get_column_count()
        self.d_table = dict((x, {}) for x, y in self.n_columns)
        self.table_dict()
        self.query_dict = self.update_query_cost()
        #print self.d_table
        #self.t_tables=tuple([i[1]+1 for i in self.n_columns])
        #low = np.zeros(len(self.t_tables), dtype=int)
        #high =  np.array(self.t_tables, dtype=int) - np.ones(len(self.t_tables), dtype=int)
        self.action_space = spaces.Discrete(self.col_len)
        self.observation_space = spaces.Box(0, 1, shape=(2*self.col_len,), dtype=int)
        self.state = None
        self.index_count = INDEX_LIMIT
        self.index_list = np.array([], dtype=int)
        self.cost = None
        self.query_cost = None

    def step(self, action):
        print("Action: %s",self.t_columns[action])
        if action not in self.index_list:
           self.db.create_index(self.t_columns[action])
        self.index_list = np.append(self.index_list,[action])
        #print self.index_list
        print self.cost
        print self.query_cost
        #print("count",self.index_count)
        if self.index_count == 0:
           done = True
        else:
           done = False
        self.index_count = self.index_count - 1
        cost = max((self.query_cost/self.cost)-1,0)
        reward = 0 if cost <= 0 else 1
        self.state = self.get_state()
        #print "next_state",self.state
        info = {}
        return self.state, reward, done, info

    def reset(self):
        self.index_count = INDEX_LIMIT
        self.index_list = np.array([], dtype=int)
        self.db.clear_index()
        np.random.shuffle(self.queries)
        self.done = False
        state = self.get_state()
        return state

    def render(self, mode='human'):
        print("Rendering")
        return 1

    def table_dict(self):
        i=0
        for x,y in self.t_columns:
            self.d_table[x][y]=i
            i=i+1

    def update_query_cost(self):
        t_queries = {}
        for i in range(self.query_len):
            self.db.state(self.queries[i])
            data = json.loads(self.db.get_query_cost(self.queries[i])[0][0])
            cost = data["query_block"]["cost_info"]['query_cost']
            t_queries[i] = {"query":self.queries[i],
                            "cost":float(cost) }
        return t_queries

    def calculate_reward(self):
        self.db.state(self.queries[i])

    def get_state(self):
        n = random.randint(0,self.query_len-1)
        query = self.query_dict[n]["query"]
        self.db.state(query)
        data = json.loads(self.db.get_query_cost(query)[0][0])
        self.cost = float(data["query_block"]["cost_info"]['query_cost'])
        self.query_cost = self.query_dict[n]["cost"]
        table = re.search(r"FROM\s(.*)WHERE", query).groups()[0].strip()
        #print "table", table
        sub = re.search(r"WHERE\s(.*)", query).groups()[0]
        col = re.sub(" \d+|AND|>|=|<|.\d+|ORDER\s(.*)|FOR\s(.*)|\'.*\'|\".*\"|;", " ", sub).split()
        #print col
        s=np.array([self.d_table[table][x] for x in col])
        index_array = np.append(s,self.index_list+self.col_len)
        #print index_array
        #print "column", col
        input_state = np.zeros(2*self.col_len, dtype=int)
        np.put(input_state,index_array,np.ones(len(index_array), dtype=int))
        return input_state
