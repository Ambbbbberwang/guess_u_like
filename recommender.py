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
                                    schema = 'user_id INT, book_id INT, is_read INT, rating FLOAT, is_reviewed INT')
        return df
    elif path=='users':
        df=spark.read.csv('hdfs:/user/bm106/pub/goodreads/user_id_map.csv', header = True, 
                                    schema = 'user_id_csv INT, user_id STRING')
        return df
    elif path=='books':
        df=spark.read.csv('hdfs:/user/bm106/pub/goodreads/book_id_map.csv', header = True, 
                                    schema = 'book_id_csv INT, book_id STRING')
        return df
    
# Data splitting and subsampling
def data_prep(spark, spark_df, pq_path, fraction=0.01, seed=42, savepq=False, filter_num=10):
    '''
    spark: spark
    spark_df: spark object
    fraction: decimal percentage of users to retrieve (i.e. 0.01, 0.05, 0.25)
    seed: set random seed for reproducibility
    savepq: if we need to process the csv, prep the data and save parquet
    pq_path: save and/or read from path (i.e. 'hdfs:/user/eac721/onepct_int.parquet')

    returns records object with random, specified subset of users
    '''

    if savepq == True:

        # remove the users with only a low number of interactions
        # https://stackoverflow.com/questions/51063624/whats-the-equivalent-of-pandas-value-counts-in-pyspark
        # https://stackoverflow.com/questions/49301373/pyspark-filter-dataframe-based-on-multiple-conditions
        import pyspark.sql.functions as f
        from pyspark.sql import Window

        w= Window.partitionBy('user_id')
        # get the number of interactions for all users - we should probably remove this column later
        spark_df=spark_df.select('user_id', 'book_id', 'is_read', 'rating', 'is_reviewed', f.count('user_id').over(w).alias('n_int')).sort('user_id')
        spark_df=spark_df.filter(spark_df.n_int>int(filter_num))
        #print((users.count(), len(users.columns)))

        # Downsampling should follow similar logic to partitioning: don't downsample interactions directly. Instead, sample a percentage of users, and take all of their interactions to make a miniature version of the data.
        #false = without replacement
        #df.sample(false ,fraction,seed)
        users=spark_df.select('user_id').distinct()
        user_samp=users.sample(False, fraction=fraction, seed=seed)

        # attn: change this to spark workflow
        #temp=temp.toPandas().iloc[:,0]
        #temp=temp.iloc[:,0]
        #temp=temp.tolist()
        
        #records=spark_df[spark_df['user_id'].isin(temp)]
        records=spark_df.where(f.col('user_id').isin(user_samp))
        print('Selected %f percent of users', records.select('user_id').distinct().count()/spark_df.select('user_id').distinct().count())

        records.write.parquet(pq_path)

    records_pq = spark.read.parquet(pq_path)

    return records_pq

# train/val, test split (60/20/20 by user_id)
def train_val_test_split(spark, records_pq, seed=42):

    # number of distinct users for checking
    print(records_pq.select('user_id').distinct().count())

    # Select 60% of users (and all of their interactions) to form the training setself.
    # Select 20% of users to form the validation set. 
    users=records_pq.select('user_id').distinct()
    temp=users.sample(False, fraction=0.6, seed=seed)
    # attn: change this to spark workflow
    temp=temp.toPandas().iloc[:,0]
    temp=temp.tolist()
    train=records_pq[records_pq['user_id'].isin(temp)].toPandas() # all interactions
    test_val=records_pq[~records_pq['user_id'].isin(temp)]

    # split test (20%), val (20%), putting half back into training set
    users=test_val.select('user_id').distinct()
    temp=users.sample(False, fraction=0.5, seed=seed)
    # attn: change this to spark workflow
    temp=temp.toPandas().iloc[:,0]
    temp=temp.tolist()
    test=test_val[test_val['user_id'].isin(temp)].toPandas()
    val=test_val[~test_val['user_id'].isin(temp)].toPandas()

    import pandas as pd

    # split test into 2 dfs: test and training interactions for all users 
    # note this excludes users with one interaction right now - should they be subset first?
    temp=test.groupby('user_id').apply(lambda x: x.sample(frac=0.5)).reset_index(drop=True)
    keys = list(temp.columns.values) 
    i1 = test.set_index(keys).index
    i2 = temp.set_index(keys).index
    test_train = test[~i1.isin(i2)]
    test = temp

    temp=val.groupby('user_id').apply(lambda x: x.sample(frac=0.5)).reset_index(drop=True)
    keys = list(temp.columns.values) 
    i1 = val.set_index(keys).index
    i2 = temp.set_index(keys).index
    val_train = val[~i1.isin(i2)]
    val = temp

    # https://stackoverflow.com/questions/54797508/how-to-generate-a-train-test-split-based-on-a-group-id
    train=pd.concat([train, val_train, test_train], axis=0)
    
    
    train=spark.createDataFrame(train, schema = 'user_id INT, book_id INT, is_read INT, rating FLOAT, is_reviewed INT')
    val=spark.createDataFrame(val, schema = 'user_id INT, book_id INT, is_read INT, rating FLOAT, is_reviewed INT')
    test=spark.createDataFrame(test, schema = 'user_id INT, book_id INT, is_read INT, rating FLOAT, is_reviewed INT')

    # check for each dataset to make sure the split works
    print(train.select('user_id').distinct().count())
    print(val.select('user_id').distinct().count())
    print(test.select('user_id').distinct().count())

    return train, val, test


### NEXT STEPS ###

# [x] (1) Convert to parquet and write files 
# [] (2) Convert wf from pandas to pyspark
# [] (3) Any items not observed during training (i.e., which have no interactions in the training set, or in the observed portion of the validation and test users), can be omitted unless you're implementing cold-start recommendation as an extension.
# [] (4) In general, users with few interactions (say, fewer than 10) may not provide sufficient data for evaluation, especially after partitioning their observations into train/test. You may discard these users from the experiment, but document your exact steps in the report.
        # DOCUMENT HERE - started by removing 10 interactions

# [] (5) Implement basic recsys: pyspark.ml.recommendation module

# [] (6) Tune HP: rank, lambda

# [] (7) Evaluate - Evaluations should be based on predicted top 500 items for each user.
        # metrics: should we use AUC, avg. precicion, reciprocal rank?

# [] (8) Main 

#def main():

#import recommender
#interactions=recommender.data_read(spark, 'interactions')
#records_pq=recommender.data_prep(spark, interactions, 0.01, 42, True, 1)
#records=recommender.data_prep(spark, interactions, 0.01, 42, False, 1)

