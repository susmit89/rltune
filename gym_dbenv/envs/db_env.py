import gym
from gym import error, spaces, utils
from client import DBENGINE
from itertools import cycle
import numpy as np
import random
import re
import json

INDEX_LIMIT = 20

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
        lst = range(len(self.query_dict.keys()))
        self.query_len = cycle(lst)
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
        self.n_query = None


    def step(self, action):
        index_name = "rl_" + self.t_columns[action][0]+"_"+ self.t_columns[action][1]
        print "Index set on ", self.t_columns[action][0], " for ",  self.t_columns[action][1]
        if action not in self.index_list:
           self.db.create_index(self.t_columns[action],index_name)
        self.index_list = np.append(self.index_list,[action])
        self.db.state(self.query)
        self.query_cost = self.query_dict[self.n_query]["cost"]
        data = json.loads(self.db.get_query_cost(self.query)[0][0])
        self.cost = float(data["query_block"]["cost_info"]['query_cost'])
        key = False
        try:
            print  "---------------", data["query_block"]["table"]["possible_keys"], "\n"
            if index_name in data["query_block"]["table"]["possible_keys"]:
                key = True
                print("---------------found key")
        except:
            pass
        try:
            print  "---------------", data["query_block"]["ordering_operation"]["table"]["possible_keys"], "\n"
            if index_name in data["query_block"]["ordering_operation"]["table"]["possible_keys"]:
                key = True
                print("---------------found key")
        except:
            pass
        #print self.cost
        #print self.query_cost
        #print("count",self.index_count)
        if self.index_count == 0:
           done = True
        else:
           done = False
        self.index_count = self.index_count - 1
        cost = max((self.query_cost/self.cost)-1,0)
        reward = 0
        if cost > 0 and key == True:
            reward = 10
        self.state = self.get_state()
        #print "next_state",self.state
        info = {}
        print "step=",self.index_count,"\t\t reward=",reward, "\tquery cost=",self.query_cost, "\tcost=", self.cost
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
        n=0
        for i in range(self.query_len):
            self.db.state(self.queries[i])
            data = json.loads(self.db.get_query_cost(self.queries[i])[0][0])
            try:
             cost = data["query_block"]["cost_info"]['query_cost']
             t_queries[n] = {"query":self.queries[i],
                            "cost":float(cost) }
             n=n+1
            except:
                print("Error in",self.queries[i])
        return t_queries

    def calculate_reward(self):
        self.db.state(self.queries[i])

    def get_state(self):
        self.n_query = next(self.query_len)

        self.query = self.query_dict[self.n_query]["query"]
        table = re.search(r"FROM\s(.*)WHERE", self.query).groups()[0].strip()
        #print "table", table
        sub = re.search(r"WHERE\s(.*)", self.query).groups()[0]
        col = re.sub(" \d+|AND|>|=|<|.\d+|ORDER\s(.*)|FOR\s(.*)|\'.*\'|\".*\"|;", " ", sub).split()
        #print col
        s=np.array([self.d_table[table][x] for x in col])
        index_array = np.append(s,self.index_list+self.col_len)
        print index_array
        #print "column", col
        input_state = np.ones(2*self.col_len, dtype=int) * -1
        np.put(input_state,index_array,np.ones(len(index_array), dtype=int))
        return input_state
