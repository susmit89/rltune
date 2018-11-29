

import pymysql
import os, json, re
from random import choice
from string import ascii_lowercase
import numpy as np



### GET Parameters
#cur.execute("SHOW VARIABLES")
#rows  = cur.fetchall()

### GET Version
#cur.execute("SELECT @@GLOBAL.version;")
#rows  = cur.fetchall()

### GET Metrics
#cur.execute("SHOW STATUS")
#rows  = cur.fetchall()

TABLE = os.environ.get('SQLTABLE')

class DBENGINE():

    def __init__(self):
        with open(os.environ.get('SQLQUERY'), 'r') as myfile:
             self.data=myfile.read()
        con = pymysql.connect(host=os.environ.get('MYSQL_HOST'),
                              port=int(os.environ.get('MYSQL_PORT')),
                              user=os.environ.get('MYSQL_USER'),
                              password=os.environ.get('MYSQL_PASS'),
                              db=os.environ.get('MYSQL_DB'))
        self.cur = con.cursor()
        self.index_table = []
        self.indexed_column = []

    def state(self, query):
        self.cur.execute(query)
        return self.cur.fetchall()

    def column_schema(self):
        self.cur.execute("SELECT TABLE_NAME, COLUMN_NAME  FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = \""+os.environ.get('MYSQL_DB')+"\" " )
        #+"AND  TABLE_NAME = \""+TABLE+"\"")
        return self.cur.fetchall()

    def index_schema(self):
        self.cur.execute("SELECT TABLE_NAME, COLUMN_NAME  FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = \""+os.environ.get('MYSQL_DB')+"\" " )
        #+"AND  TABLE_NAME = \""+TABLE+"\"")
        return self.cur.fetchall()

    def get_column_count(self):
        self.cur.execute("SELECT TABLE_NAME, count(*)  FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = \""+os.environ.get('MYSQL_DB')+"\" " + "GROUP BY TABLE_NAME HAVING COUNT(*)>1;" )
        #+" AND  TABLE_NAME = \""+TABLE+"\" GROUP BY TABLE_NAME HAVING COUNT(*)>1 ;")
        return self.cur.fetchall()

    def get_query_cost(self,query, index_name):
        self.cur.execute("explain format=JSON "+ query)
        data = json.loads(self.cur.fetchall()[0][0])
        cost = float(data["query_block"]["cost_info"]['query_cost'])
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
        return (cost, key)

    def create_index(self, index, index_name):
        self.cur.execute("CREATE INDEX "+index_name+" ON "+index[0]+" ("+index[1]+")")
        self.index_table.append((index_name,index[0],index[1]))

    def clear_index(self):
        for i in self.index_table:
            self.cur.execute("ALTER TABLE "+i[1]+" DROP INDEX "+i[0]+";")
        self.index_table = []

    def get_query_workload(self):
        self.queries = np.array(self.data.splitlines())
        return self.queries

    def update_query_cost(self):
        t_queries = {}
        n=0
        self.query_len = self.get_query_workload()
        print self.query_len
        for i in range(len(self.query_len)):
            #self.state(self.query_len[i])
            #data = json.loads(self.db.get_query_cost(self.queries[i])[0][0])
            try:
             query = self.query_len[i]
             print self.get_query_cost(self.query_len[i], None)
             (cost,_) = self.get_query_cost(query, None)
             table = re.search(r"FROM\s(.*)WHERE|from\s(.*)where", query).groups()[0].strip()
             print "table", table
             sub = list(re.search(r"WHERE\s(.*)|where\s(.*)", query).groups())
             str = " ".join([x for x in sub if x is not None])
             col = re.sub("'([^']*)'|\"([^']*)\"|\d+|AND|>|=|<|.\d+|ORDER\s(.*)|FOR\s(.*)|limit\s(.*)|;", " ", str).split()
             tc = [(table,x) for x in col]
             t_queries[n] = {"query":query,
                            "cost":float(cost),"columns":tc }
             n=n+1
            except Exception as e:
                print(e)
                print("Error in",self.queries[i])
        print t_queries
        return t_queries
