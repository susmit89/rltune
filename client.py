

import pymysql
con = pymysql.connect(host='18.236.158.133',
                      port=9000,
                      user='root',
                      password='root123')

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
