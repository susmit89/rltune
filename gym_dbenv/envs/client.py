

import pymysql
import os





### GET Parameters
#cur.execute("SHOW VARIABLES")
#rows  = cur.fetchall()

### GET Version
#cur.execute("SELECT @@GLOBAL.version;")
#rows  = cur.fetchall()

### GET Metrics
#cur.execute("SHOW STATUS")
#rows  = cur.fetchall()


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

    def state(self, query):
        self.cur.execute(query)
        return self.cur.fetchall()

    def column_schema(self):
        self.cur.execute("SELECT TABLE_NAME, COLUMN_NAME  FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = \""+os.environ.get('MYSQL_DB')+"\"")
        return self.cur.fetchall()

    def index_schema(self):
        self.cur.execute("SELECT TABLE_NAME, COLUMN_NAME  FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = \""+os.environ.get('MYSQL_DB')+"\"")
        return self.cur.fetchall()

    def get_column_count(self):
        self.cur.execute("SELECT TABLE_NAME, count(*)  FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = \""+os.environ.get('MYSQL_DB')+"\"  GROUP BY TABLE_NAME HAVING COUNT(*)>1 ;")
        return self.cur.fetchall()

    def get_query_cost(self,query):
        self.cur.execute("explain format=JSON "+ query)
        return self.cur.fetchall()

    def create_index(self, index_name, index):
        self.cur.execute("CREATE INDEX "+index_name+" ON "+index[0]+" ("+index[1]+")")
        self.index_table.append((index_name,index[0]))
        print self.index_table
        pass

    def clear_index(self, table, index):
        pass

    def get_query_workload(self):
        return self.data
