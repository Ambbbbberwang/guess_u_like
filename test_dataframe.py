from pyspark import Row

test_df = spark.createDataFrame([(1, [Row(a=1,b=2), Row(a=3,b=4)]), (2, [Row(a=3,b=4),Row(a=0,b=5)]), (3, [Row(a=-3,b=2),Row(a=13,b=40)])],("id", "an_array"))


test_df = test_df.select('id','an_array',posexplode('an_array')).drop('pos').drop('an_array')

test_df.show()

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

---> Still need to do this part:

id|  a | b|
1 |  1 | 2|
1 |  3 | 4|
...






'''

