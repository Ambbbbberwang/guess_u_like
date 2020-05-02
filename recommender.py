#!/usr/bin/env python

#read in data
def data_read(spark, path):
    '''
    spark: spark
    file_path: 
        - interactions path: hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv
        - user_id path: hdfs:/user/bm106/pub/goodreads/user_id_map.csv
        - book_id path: hdfs:/user/bm106/pub/goodreads/book_id_map.csv
    which_csv: 'interactions', 'users', 'books'

    returns spark dataframe object with specified schema
    '''
    if path=='interactions':
        df=spark.read.csv('hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv', header = True, 
                                    schema = 'user_id STRING, book_id STRING, is_read INT, rating FLOAT, is_reviewed INT')
        return df
    elif path=='users':
        df=spark.read.csv('hdfs:/user/bm106/pub/goodreads/user_id_map.csv', header = True, 
                                    schema = 'user_id_csv STRING, user_id STRING')
        return df
    elif path=='books':
        df=spark.read.csv('hdfs:/user/bm106/pub/goodreads/book_id_map.csv', header = True, 
                                    schema = 'book_id_csv STRING, book_id STRING')
        return df
    
# Data subsampling
def data_prep(spark, spark_df, pq_path='hdfs:/user/eac721/onepct_int.parquet', fraction=0.01, seed=42, savepq=False, filter_num=10):
    '''
    spark: spark
    spark_df: spark df, output from date_read
    fraction: decimal (%) of users to retrieve (i.e. 0.01, 0.05, 0.25)
    seed: set random seed for reproducibility
    savepq: whether we need to process csv file; if True, prep the data and save parquet
    pq_path: save and/or read from path (i.e. 'hdfs:/user/eac721/onepct_int.parquet')

    returns records object with random, specified subset of users
    '''

    if savepq == True:

        # Recommender constraint: remove the users with only a low number of interactions

        import pyspark.sql.functions as f
        from pyspark.sql import Window

        w= Window.partitionBy('user_id')
        # Add a column with the number of interactions for all users 
        # note: we should rm this column using drop command
        spark_df=spark_df.select('user_id', 'book_id', 'is_read', 'rating', 'is_reviewed', f.count('user_id').over(w).alias('n_int')).sort('user_id')
        spark_df=spark_df.filter(spark_df.n_int>int(filter_num))
        spark_df=spark_df.drop('n_int')
        #test the drop works:
        #spark_df.createOrReplaceTempView('spark_df') #test before filter
        #small = spark.sql('SELECT * FROM spark_df WHERE n_int <= 10')
        #small.count() #460348

        #Original interaction dataset 228,648,342 rows
        #After dropping users with iteractions less than 10, remains rows 228,187,994
        #spark_df.show()
 
        # downsampling: sample a percentage of users, and take all of their interactions to make a miniature version of the data.
        users=spark_df.select('user_id').distinct() 
        user_samp=users.sample(False, fraction=fraction, seed=seed) #77726
        #user_samp.show()
        

        # inner join: keep only the randomly sampled users and all their interactions (based on percentage specified)
        records=spark_df.join(user_samp, ['user_id']) #2387376
    

        # check that this matches the desired percentage
        #print(records.select('user_id').distinct().count())  #7678
        #print(spark_df.select('user_id').distinct().count()) #766717 

        # write to parquet format
        # note: this will fail if the path already exists 
        # remove the file with "hadoop fs -rm -r onepct_int.parquet"
        records.orderBy('user_id').write.parquet(pq_path)

    #return records

def rand_samp(df):

    users = df.select('user_id').distinct()

    test = sqlContext.createDataFrame(sc.emptyRDD(), df.schema)
    val = sqlContext.createDataFrame(sc.emptyRDD(), df.schema)  

    for u in users: 
        temp = df.filter(df['user_id']==u)
        temp1, temp2 = temp.randomSplit([0.5, 0.5], seed = 42)
        test = test.union(temp1)
        val = val.union(temp2)

    return test, val

# Data splitting
def train_val_test_split(spark, records_path='hdfs:/user/eac721/onepct_int.parquet', seed=42):

    '''
    # This function takes the following splitting procedure: 
    # Select 60% of users (and all of their interactions).
    # Select 20% of users to form the validation set (half interactions for training, half in validation). 
    # Select 20% of users to form the test set (same as validation).

    spark: spark
    records_pq: spark df output from data_prep function
    seed: random seed

    returns train, val, test data
    '''

    records_pq = spark.read.parquet(records_path)

    records_pq = records_pq.withColumn('user_id', records_pq['user_id'].cast('string'))
    # number of distinct users for checking
    #print(records_pq.select('user_id').distinct().count())

    # find the unique users:
    #users=records_pq.select('user_id').distinct()
    #print(users.count()) #7711

    # sample the 60% and all interactions to form the training set and remaining set (test and val)
    users=records_pq.select('user_id').distinct()
    user_samp=users.sample(False, fraction=0.6, seed=seed) 
    train=user_samp.join(records_pq, ['user_id'])
    test_val=records_pq.join(user_samp, ['user_id'], 'left_anti') 
    #print(train.select('user_id').distinct().count())  #4591
    print('Number of validation users',test_val.select('user_id').distinct().count()) #3120


    # split the remaining set into 50/50 by users' interactions
    #print(test_val.groupBy('user_id').count().orderBy('user_id').show())
    #users2=test_val.select('user_id').distinct().collect()
    #frac = dict((u.user_id, 0.5) for u in users2)
    df_count = test_val.groupBy('user_id').count().sort('user_id')
    frac = dict(df_count.rdd.map(lambda x:(x['user_id'], 0.5)).collect())
    #print(len(frac))
    test_val_train=test_val.sampleBy('user_id', fractions=frac, seed=seed)
    print('Number of validation users to training set (should equal to all validation users)',test_val_train.select('user_id').distinct().count())

    test_val=test_val.join(test_val_train, ['user_id', 'book_id'], 'left_anti') 
    
    #print(test_val.groupBy('user_id').count().orderBy('user_id').show())
    # add training 50% back to train
    train=train.union(test_val_train) 
    #print(train.select('user_id').distinct().count())
    #print(test_val.select('user_id').distinct().count())

    #####test all users in test_val are in train#####
    train_user = train.select('user_id').distinct().collect()
    val_user = test_val.select('user_id').distinct().collect()
    for u in val_user:
        if u not in train_user:
            print('I am not included in train!! (user_id)',u)


   # split the test_val set into test (20%), val (20%) by user
    users3=test_val.select('user_id').distinct()
    user_samp=users3.sample(False, fraction=0.5, seed=seed)
    test=user_samp.join(test_val, ['user_id']) 
    val=test_val.join(user_samp, ['user_id'], 'left_anti')
    #print(val.select('user_id').distinct().count())
    #print(test.select('user_id').distinct().count())

    # remove items that are not in training from all three datasets
    # find items in val and/or test that are not in train
    itemsv=val.select('book_id').distinct()
    itemst=test.select('book_id').distinct()
    items_testval=itemsv.union(itemst).distinct()
    #print(items_testval.orderBy('book_id').show()) 
    items_train=train.select('book_id').distinct()
    #print(items_train.orderBy('book_id').show())
    items_rm=items_testval.join(items_train, ['book_id'], 'leftanti')
    #print(items_rm.orderBy('book_id').show())

    #print(train.groupBy('book_id').count().orderBy('book_id').show())
    #print(val.groupBy('book_id').count().orderBy('book_id').show())
    #print(test.groupBy('book_id').count().orderBy('book_id').show())
    train=train.join(items_rm, ['book_id'], 'left_anti')
    val=val.join(items_rm, ['book_id'], 'left_anti')
    test=test.join(items_rm, ['book_id'], 'left_anti')
    #print(train.groupBy('book_id').count().orderBy('book_id').show())
    #print(val.groupBy('book_id').count().orderBy('book_id').show())
    #print(test.groupBy('book_id').count().orderBy('book_id').show())
    
    # check for each dataset to make sure the split works
    #print(train.select('user_id').distinct().count())
    #print(val.select('user_id').distinct().count())
    #print(test.select('user_id').distinct().count())

    # subset the data for modeling
    train = train.select("user_id","book_id","rating")
    val = val.select("user_id","book_id","rating")
    test = test.select("user_id","book_id","rating")

    return train, val, test 

# Fitting model
def recsys_fit(train, val, test, ranks=[10], regParams=[0.1]):

    '''
    This function fits the recommender system using train to train, val to hp tune, and test to evaluate
    
    train: Input training data to fit the model
    val: Input validation data to tune the model
    test: Input test data to evaluate the model
    ranks: List of ranks to be tuned (default = [10])
    regParams: List of regParams to be tuned (default = [0.1])
    
    returns the optimal ALS model 

    References:
    # https://spark.apache.org/docs/latest/api/python/pyspark.ml.html#module-pyspark.ml.recommendation
    # https://www.kaggle.com/vchulski/tutorial-collaborative-filtering-with-pyspark
    # https://spark.apache.org/docs/2.2.0/ml-collaborative-filtering.html
    # https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3175648861028866/48824497172554/657465297935335/latest.html    
    
    '''

    from pyspark.ml.recommendation import ALS #, Rating
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.sql import functions as f
    from pyspark.sql.types import DoubleType

    # Build the recommendation model using ALS on the training data
    als = ALS(maxIter=10, regParam=0.01, 
          userCol="user_id", itemCol="book_id", ratingCol="rating",
          coldStartStrategy="drop",
          implicitPrefs=False)
    #model = als.fit(train)
    #predictions = model.transform(val)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    
    # hyperparameter tuning: grid serach for rank, lambda using validation set
    print('Running grid search:')
    #ranks = [5, 10, 20], regParams = [0.01, 0.05, 0.1]
    # Set up the model and error checking
    models = [[0]*len(ranks)]*len(regParams)
    errors = [[0]*len(ranks)]*len(regParams)

    # Initialize the errors and hps
    err = 0
    min_error = float('inf')
    best_rank = -1
    i = 0

    # For each combo of params, fit the model and create a prediction using val data
    for regParam in regParams:
      j = 0
      for rank in ranks:
        
        als.setParams(rank = rank, regParam = regParam)
        this_model = als.fit(train)
        predict_df = this_model.transform(val)

        predicted_ratings_df = predict_df.filter(predict_df.prediction != float('nan'))
        predicted_ratings_df = predicted_ratings_df.withColumn("prediction", f.abs(f.round(predicted_ratings_df["prediction"],0)))
        
        this_error = evaluator.evaluate(predicted_ratings_df)
        errors[i][j] = this_error
        models[i][j] = this_model
        
        print('For rank %s, regularization parameter %s the RMSE is %s' % (rank, regParam, this_error))
        
        if this_error < min_error:
          min_error = this_error
          best_params = [i, j]

        j += 1
      i += 1

    als.setRegParam(regParams[best_params[0]])
    als.setRank(ranks[best_params[1]])
    best_model = models[best_params[0]][best_params[1]]
    print('The best model was trained with regularization parameter %s and rank %s' % (regParams[best_params[0]], ranks[best_params[1]]))

    print('Fitting the champion model:')
    test_df = test.withColumn("rating", test["rating"].cast(DoubleType()))
    predict_df = best_model.transform(test_df)
    # TO DO: pick only the top 500 for each user
    predicted_test_df = predict_df.filter(predict_df.prediction != float('nan'))
    # Round floats to whole numbers to compare
    predicted_test_df = predicted_test_df.withColumn("prediction", f.abs(f.round(predicted_test_df["prediction"],0)))
    # Run the previously created RMSE evaluator, reg_eval, on the predicted_test_df DataFrame
    test_RMSE = evaluator.evaluate(predicted_test_df)
    print('The champion model had a RMSE on the test set of {0}'.format(test_RMSE))

    return best_model # what do we want to return here?



# DRAFT
"""
def recsys (train, val, test, ranks = [10, 15, 20], regParams = [0.005, 0.01, 0.15], 
            maxIters = [10]):
    
    from pyspark.ml.recommendation import ALS #, Rating
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.sql import functions as f
    from pyspark.sql.types import DoubleType
    #from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
    #from hypopt import GridSearch
    #from sklearn.metrics import make_scorer
    import itertools
    import numpy as np

    # Build the recommendation model using ALS on the training data
    als = ALS(userCol="user_id", itemCol="book_id", ratingCol="rating",
              coldStartStrategy="drop", implicitPrefs=False, seed = 42)
    
    #paramMap = ParamGridBuilder() \
    #                .addGrid(als.rank, ranks) \
    #                .addGrid(als.maxIter, maxIters) \
    #                .addGrid(als.regParam, regParams) \
    #                .build()
    
    param_list=[ranks,regParams, maxIters]
    param_grid=list(iterrools.product(*param_list))
    
    models = np.zeros([len(ranks), len(regParams), len(maxIters)])
    errors = np.zeros([len(ranks), len(regParams), len(maxIters)])
    
    err = 0
    min_error = float('inf')
    best_rank = -1
    i = 0
    
    print('Running grid search:')
    for params in param_grid:
        print('Try :', params)
        als.setParams(rank=params[0], regParam=params[1], maxIter=params[2])
        cur_model = als.fit(train)
        predict_df = this_model.transform(val)
        
        cur_model.recommendForAllUsers(500)
        
        
        # To Be Continuted...
"""

# Extension 1: 
# Exploration: use the learned representation to develop a visualization of the items and users, 
# e.g., using T-SNE or UMAP. The visualization should somehow integrate additional information 
# (features, metadata, or genre tags) to illustrate how items are distributed in the learned space.

# References: 
# https://www.liip.ch/en/blog/the-magic-of-tsne-for-visualizing-your-data-features
# https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
# https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1


### NEXT STEPS ###
# [x] (1) Convert to parquet and write file function 
# [x] (2) Check the splitting function for correctness
# [x] (3) Check removal of items for correctness
# [x] (4) In general, users with few interactions (say, fewer than 10) may not provide sufficient data for evaluation, especially after partitioning their observations into train/test. You may discard these users from the experiment, but document your exact steps in the report.
        # DOCUMENT HERE - started by removing users with fewer than 10 interactions in the very beginning of the script
                        # NOTE: this is a parameter we can tune later

# [x] (5) Implement basic recsys: pyspark.ml.recommendation module

# [x] (6) Tune HP: rank, lambda
# [o]      # NOTE: could improve by breaking out hp tuning into a function
           # REMINDER: high rank --> overfitting; low rank --> underfitting 

# [x] (7) Evaluate - RSME
           # NOTE: Evaluations should be based on predicted top 500 items for each user.
# [o]      # NOTE: improve using metrics: avg. precision, reciprocal rank for validation

# [o] (8) Main?

# [o] (9) Extension 1

# [o] (10) Extension 2

#main()


#import recommender
#interactions=recommender.data_read(spark, 'interactions')
#records=recommender.data_prep(spark, interactions, 'hdfs:/user/eac721/onepct_int.parquet', 0.01, 42, True, 10)
## records=recommender.data_prep(spark, interactions, 'hdfs:/user/eac721/onepct_int.parquet', 0.01, 42, False, 10)
#train, val, test = recommender.train_val_test_split(spark,records)
#model = recommender.recsys_fit(train, val, test)


# # Generate top 10 movie recommendations for a specified set of users
#users = ratings.select(als.getUserCol()).distinct().limit(3)
#userSubsetRecs = model.recommendForUserSubset(users, 10)
# Generate top 10 user recommendations for a specified set of movies
#movies = ratings.select(als.getItemCol()).distinct().limit(3)
#movieSubSetRecs = model.recommendForItemSubset(movies, 10)