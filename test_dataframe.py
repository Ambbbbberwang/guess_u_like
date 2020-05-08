from pyspark import Row
from pyspark.sql.functions import *



test_df = spark.createDataFrame([(1, [Row(a=1,b=2), Row(a=3,b=4)]), (2, [Row(a=3,b=4),Row(a=0,b=5)]), (3, [Row(a=-3,b=2),Row(a=13,b=40)])],("id", "an_array"))


test_df = test_df.select('id','an_array',posexplode('an_array')).drop('pos').drop('an_array')



test_df.select('id',f.expr('col.a'),f.expr('col.b')).show()



'''
+---+-------------------+
| id|           an_array|
+---+-------------------+
|  1|   [[1, 2], [3, 4]]|
|  2|   [[3, 4], [0, 5]]|
|  3|[[-3, 2], [13, 40]]|
+---+-------------------+



---> DONE this part:


+---+--------+
| id|     col|
+---+--------+
|  1|  [1, 2]|
|  1|  [3, 4]|
|  2|  [3, 4]|
|  2|  [0, 5]|
|  3| [-3, 2]|
|  3|[13, 40]|
+---+--------+

---> DONE this part:

id|  a | b|
1 |  1 | 2|
1 |  3 | 4|
...


'''

###test for cosine similarity calculation of two spark dataframe###
from pyspark.sql import functions as F
df1 = sqlContext.createDataFrame([(1,2,3),(2,-1,2)],('CustomerID','CustomerValue'))
df2 = sqlContext.createDataFrame([(3,[0,2,3])],('id','features'))


kvs = F.explode(F.array([F.struct(F.lit('features').alias('key'), F.column('features').alias('value'))])).alias('kvs')

dft1 = (df1.select(['id', kvs]).select('id', F.column('kvs.name').alias('column_name'), F.column('kvs.value').alias('column_value')))


