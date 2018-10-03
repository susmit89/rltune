

import pymysql
import os
con = pymysql.connect(host=os.environ.get('MYSQL_HOST'),
                      port=os.environ.get('MYSQL_PORT'),
                      user=os.environ.get('MYSQL_USER'),
                      password=os.environ.get('MYSQL_PASS'))

cur = con.cursor()


### GET Parameters
cur.execute("SHOW VARIABLES")
rows  = cur.fetchall()

### GET Version
cur.execute("SELECT @@GLOBAL.version;")
rows  = cur.fetchall()

### GET Metrics
cur.execute("SHOW STATUS")
rows  = cur.fetchall()
