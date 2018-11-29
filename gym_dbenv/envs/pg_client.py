

import psycopg2
import os, json
from random import choice
from string import ascii_lowercase




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

class PGDB():

    def __init__(self):
        with open(os.environ.get('SQLQUERY'), 'r') as myfile:
             self.data=myfile.read()
        con = psycopg2.connect("dbname='"+os.environ.get('MYSQL_DB')+
                "' user='"+os.environ.get('MYSQL_USER')+
                "' host='"+os.environ.get('MYSQL_HOST')+"' port='"+os.environ.get('MYSQL_PORT')+
                "' password='"+os.environ.get('MYSQL_PASS')+"'")
        self.cur = con.cursor()
        self.index_table = []
        self.indexed_column = []

    def state(self, query):
        self.cur.execute(query)
        return self.cur.fetchall()

    def column_schema(self):
        self.cur.execute("SELECT TABLE_NAME, COLUMN_NAME  FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = \'"+os.environ.get('MYSQL_DB')+"\' ;" )
        #+"AND  TABLE_NAME = \""+TABLE+"\"")
        return self.cur.fetchall()

    def index_schema(self):
        self.cur.execute("SELECT TABLE_NAME, COLUMN_NAME  FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = \'"+os.environ.get('MYSQL_DB')+"\' ;" )
        #+"AND  TABLE_NAME = \""+TABLE+"\"")
        return self.cur.fetchall()

    def get_column_count(self):
        self.cur.execute("SELECT TABLE_NAME, count(*)  FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = \'"+os.environ.get('MYSQL_DB')+"\' " + "GROUP BY TABLE_NAME HAVING COUNT(*)>1;" )
        #+" AND  TABLE_NAME = \""+TABLE+"\" GROUP BY TABLE_NAME HAVING COUNT(*)>1 ;")
        return self.cur.fetchall()

    def get_query_cost(self,query,index_name):
        self.cur.execute("explain (format json) "+ query)
        #print self.cur.fetchall()[0][0][0]
        data = self.cur.fetchall()[0][0][0]
        cost = float(data["Plan"]["Total Cost"])
        key = False
        try:
            print  "---------------", data[0][0][0]["Plan"]["Index Name"], "\n"
            if index_name == data[0][0][0]["Plan"]["Index Name"]:
                key = True
                print("---------------found key")
        except:
            pass
        try:
            print  "---------------", data[0][0][0]["Plan"]["Plans"], "\n"
            if index_name ==  data[0][0][0]["Plan"]["Plans"][0]["Index Name"]:
                key = True
                print("---------------found key")
        except:
            pass
        print cost
        return (cost, key)

    def create_index(self, index, index_name):
        #self.cur.execute("CREATE INDEX "+index_name+" ON "+index[0]+" ("+index[1]+");")
        self.index_table.append((index_name,index[0],index[1]))

    def clear_index(self):
        for i in self.index_table:
            self.cur.execute(" DROP INDEX "+i[0]+";")
        self.index_table = []

    def get_query_workload(self):
        return self.data
