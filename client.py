

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
        con = pymysql.connect(host=os.environ.get('MYSQL_HOST'),
                              port=int(os.environ.get('MYSQL_PORT')),
                              user=os.environ.get('MYSQL_USER'),
                              password=os.environ.get('MYSQL_PASS'))
        self.cur = con.cursor()

    def state(self, query):
        self.cur.execute(query)
        return self.cur.fetchall()
