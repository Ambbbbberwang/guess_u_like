#!/usr/bin/env python

#Remove modules to Top, shared for all functions
import pyspark.sql.functions as f
from pyspark.sql import Window

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
                                    schema = 'user_id INT, book_id INT, is_read INT, rating FLOAT, is_reviewed INT')
        return df
    elif path=='users':
        df=spark.read.csv('hdfs:/user/bm106/pub/goodreads/user_id_map.csv', header = True, 
                                    schema = 'user_id_csv INT, user_id INT')
        return df
    elif path=='books':
        df=spark.read.csv('hdfs:/user/bm106/pub/goodreads/book_id_map.csv', header = True, 
                                    schema = 'book_id_csv INT, book_id INT')
        return df
    
# Data subsampling
def data_subsample(spark, spark_df, pq_path='hdfs:/user/eac721/onepct_int.parquet', fraction=0.01, seed=42, savepq=False, filter_num=10):
    '''
    spark: spark
    spark_df: spark df, output from data_read
    pq_path: the path you wish to save or load from (i.e. 'hdfs:/user/eac721/onepct_int.parquet')
    fraction: decimal of users to retrieve for subsampling (i.e. 0.01, 0.05, 0.25)
    seed: set random seed for reproducibility (default is 42)
    savepq: whether we need to process csv file; if True, prep the data and save parquet
    filter_num: the number of items a user must have in their history to be rated (i.e. 10)

    returns records object with random, specified subset of users
    '''

    if savepq == True:

        # Recommender constraint: remove the users with only a low number of interaction

        w= Window.partitionBy('user_id')
        # Add a column with the number of interactions for all users 
        # note: we should rm this column using drop command
        spark_df=spark_df.select('user_id', 'book_id', 'is_read', 'rating', 'is_reviewed', f.count('user_id').over(w).alias('n_int')).sort('user_id')
        spark_df=spark_df.filter(spark_df.n_int>int(filter_num))
        spark_df=spark_df.drop('n_int')
        
        # check: test that the drop works
        #spark_df.createOrReplaceTempView('spark_df') # originial 228,648,342 rows
        #small = spark.sql('SELECT * FROM spark_df WHERE n_int <= 10') # remains rows 228,187,994
        #small.count() #460348
 
        # downsampling: sample a percentage of users, and take all of their interactions to make a miniature version of the data.
        spark_df.createOrReplaceTempView('spark_df')
        users = spark.sql('SELECT DISTINCT user_id FROM spark_df')
        splits = users.randomSplit([fraction, 1-fraction], seed=seed)
        user_samp = splits[0]

        # inner join: keep only the randomly sampled users and all their interactions (based on percentage specified)
        records=user_samp.join(spark_df, on='user_id', how = 'inner') #2387376
    
        # check: test that this matches the desired percentage
        #print(records.select('user_id').distinct().count())  #7678
        #print(spark_df.select('user_id').distinct().count()) #766717 

        # write to parquet format
        # note: this will fail if the path already exists 
        # remove the file with "hadoop fs -rm -r onepct_int.parquet"
        records.orderBy('user_id').write.parquet(pq_path)

    else: 
        records = spark.read.parquet(pq_path)

    return records


# Data splitting
def train_val_test_split(spark, records_path='hdfs:/user/yw2115/onepct_book.parquet', seed=42):

    '''
    # This function takes the following splitting procedure: 
    # Select 60% of users (and all of their interactions).
    # Select 20% of users to form the validation set (half interactions for training, half in validation). 
    # Select 20% of users to form the test set (same as validation).

    spark: spark
    records_path: parquet file path to read from (i.e. 'hdfs:/user/eac721/onepct_int.parquet')
    seed: random seed (default = 42)

    returns train, val, test 
    '''

    # read the data_prepped file 
    records_pq = spark.read.parquet(records_path)

    # find the unique users:
    records_pq.createOrReplaceTempView('records_pq')
    users = spark.sql('SELECT DISTINCT user_id FROM records_pq')
    #print(users.select('user_id').distinct().count())

    # split 60% of the users to form the training set and remaining set (test and val)
    train_user, test_val_user = users.randomSplit([0.6, 0.4], seed=seed)

    # rejoin them back to the original parquet file on user_id
    train=train_user.join(records_pq, on='user_id', how='inner')
    test_val=test_val_user.join(records_pq, on='user_id', how='inner')
    #print(train.select('user_id').distinct().count())
    #print(test_val.select('user_id').distinct().count())

    # check: train and test_val don't have overlapping users
    train_user1 = train.select('user_id').distinct().collect()
    test_val_user1 = test_val.select('user_id').distinct().collect()
    for u in test_val_user1:
        if u in train_user1:
            print('First split: This is a problem! User in both train and val:',u)
    """
    # within the test-val set, return half the interactions of each user to the training set
    frac2 = test_val.rdd.map(lambda x: x['user_id']).distinct().map(lambda x: (x,0.5)).collectAsMap()
    test_val_kb = test_val.rdd.keyBy(lambda x: x['user_id'])
    test_val_train=test_val_kb.sampleByKey('user_id', fractions=frac2, seed=seed).map(lambda x: x[1]).toDF(test_val.columns)
    # make sure schemas match
    test_val_train = test_val_train.withColumn('is_read',test_val_train['is_read'].cast('int'))
    test_val_train = test_val_train.withColumn('is_reviewed',test_val_train['is_reviewed'].cast('int'))
    test_val_train = test_val_train.withColumn('rating',test_val_train['rating'].cast('float'))
    #print('schema of test_val_train', test_val_train.printSchema)
    #print('schema of train', train.printSchema)

    # build the true test-val set and return the rest to the training set (half of each users interactions)
    test_val=test_val.join(test_val_train, ['user_id', 'book_id'], 'left_anti') 
    train=train.union(test_val_train) 
    """
    
    # new way to derive test_val_train
    window = Window.partitionBy('user_id').orderBy('book_id')
    test_val_indexed=test_val.select('user_id','book_id','is_read','rating','is_reviewed',f.row_number().over(window).alias('row_number'))
    test_val_train = test_val_indexed.filter(test_val_indexed.row_number % 2 == 0).drop('row_number')
    test_val = test_val_indexed.filter(test_val_indexed.row_number % 2 == 1).drop('row_number')
    train = train.union(test_val_train)

    # check: test all users in test_val are in train 
    train_user = train.select('user_id').distinct().collect()
    test_val_user = test_val.select('user_id').distinct().collect()
    for u in test_val_user:
        if u not in train_user:
            print('I am not included in train!! (user_id)',u)

    # split the remaining test_val set into 50/50 by users to form the separate test and validation sets (20% of overall user-base each)
    # select the unique users
    test_val.createOrReplaceTempView('test_val')
    users = spark.sql('SELECT DISTINCT user_id FROM test_val')
    # split 50/50
    test_user, val_user = users.randomSplit([0.5, 0.5], seed=seed)
    # build the separate test and val sets
    test=test_user.join(test_val, on='user_id', how='inner')
    val=val_user.join(test_val, on='user_id', how='inner')

    test_user = test.select('user_id').distinct().collect()
    val_user = val.select('user_id').distinct().collect()
    for u in test_user:
        if u in val_user:
            print('Oops: I am in both test and val sets.',u)


    # remove items that are not in training from all three datasets
    # find items in val and/or test that are not in train
    itemsv=val.select('book_id').distinct()
    itemst=test.select('book_id').distinct()
    items_testval=itemsv.union(itemst).distinct()
    #print(items_testval.orderBy('book_id').show()) 
    items_train=train.select('book_id').distinct()
    #print(items_train.orderBy('book_id').show())
    items_rm=items_testval.join(items_train, on='book_id', how='leftanti')
    #print(items_rm.orderBy('book_id').show())

    # join the items to be removed with each train, val, test set to ensure that we don't include those items in our recommendation

    # check
    #print(train.groupBy('book_id').count().orderBy('book_id').show())
    #print(val.groupBy('book_id').count().orderBy('book_id').show())
    #print(test.groupBy('book_id').count().orderBy('book_id').show())

    train=train.join(items_rm, on='book_id', how='left_anti')
    val=val.join(items_rm, on='book_id', how='left_anti')
    test=test.join(items_rm, on='book_id', how='left_anti')

    # check
    #print(train.groupBy('book_id').count().orderBy('book_id').show())
    #print(val.groupBy('book_id').count().orderBy('book_id').show())
    #print(test.groupBy('book_id').count().orderBy('book_id').show())
    
    # final check: how many users are in each separate dataset compared to the first users count?
    #print(train.select('user_id').distinct().count())
    #print(val.select('user_id').distinct().count())
    #print(test.select('user_id').distinct().count())

    # subset the data for modeling where we need only these columns: "user_id","book_id","rating"
    train = train.select("user_id","book_id","rating")
    val = val.select("user_id","book_id","rating")
    test = test.select("user_id","book_id","rating")

    return train, val, test 
