#!/usr/bin/env python



def evaluation(model,val,metric):

	##RSME##
	from pyspark.ml.recommendation import ALS #, Rating
	from pyspark.mllib.evaluation import RegressionMetrics
	import pyspark.sql.functions as f

	als = ALS(maxIter=10, regParam=0.01, userCol="user_id", itemCol="book_id", ratingCol="rating",
	              coldStartStrategy="drop", implicitPrefs=False, seed = 42)
	model = als.fit(train)


	#all users in val
	user_val = val.select('user_id').distinct()
	#recommend top 500 books for each user in val
	val_rec = model.recommendForUserSubset(user_val,500)
	#print(val_rec.first())
	#DataFrame[user_id: int, recommendations: array<struct<book_id:int,rating:float>>]



	#####Reshape the dataframe######

	#This part is not finished yet!!
	#Need to first reshape the dataframe of val_rec to the same as val
	#with col names 'user_id', 'book_id', 'rating' | instead of 'recommendations'
	#before running the following code!!!

	#Please refer to test_dataframe.py for more test code

	#################################


	val_label = val.rdd.map(lambda r: ((r.user_id, r.book_id), r.rating))
	val_rec = val_rec.rdd.map(lambda r: ((r.user_id, r.book_id), r.rating))
	#print(val_label.first()) #((408510, 148), 0.0) 

	scoreAndLabels = val_rec.join(val_label).map(lambda tup: tup[1])

	metrics = RegressionMetrics(scoreAndLabels)
	print("RMSE = %s" % metrics.rootMeanSquaredError)









