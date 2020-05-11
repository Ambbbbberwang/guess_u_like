#!/usr/bin/env python

# rec_fit & rec_test & ranking_evaluator

from pyspark.ml.recommendation import ALS, ALSModel #, Rating
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as f
import timeit
from pyspark.sql import Window
from pyspark.mllib.evaluation import RankingMetrics
from coldstart import *
from pyspark.sql import SQLContext
from pyspark.sql import functions as f
from pyspark.sql.types import FloatType



#----------------Recommender Fit------------------------

def RecSys_fit (spark, train, val, metric = 'RMSE', seed = 42,ranks = [10, 15], 
            regParams = [0.01, 0.15], maxIters = [10]):
 
      
    # Record Start Time
    fit_start=timeit.default_timer()

    # Build the recommendation model using ALS on the training data
    als = ALS(userCol="user_id", itemCol="book_id", ratingCol="rating",
              coldStartStrategy="drop", implicitPrefs=False, seed = seed)
    
    if metric == 'RMSE':
        eval_RMSE = RegressionEvaluator(metricName="rmse", labelCol="rating", 
                                        predictionCol="prediction")
        # Initialize best_score to track min
        best_score = float('inf')
    
    elif metric in ['Precision','MAP','NDCG']:
        
        #Initialize best_score to track max
        best_score = -1*float('inf')  
    
    else:
        raise ValueError("Score metric not supported.")
    
    #models = np.zeros([len(ranks), len(regParams), len(maxIters)])
    #scores = np.zeros([len(ranks), len(regParams), len(maxIters)])
    
    #Initialize best model
    best_model = None

    #Used for print out 2 unselected ranking metrics for reference
    other2 = None
    
    print('Running grid search:')
    for i, rank in enumerate(ranks):
        for j, regParam in enumerate(regParams):
            for p, maxIter in enumerate(maxIters):
                
                # Set corresponding tuned parameters for this round & Fit
                als.setParams(rank=rank, regParam=regParam, maxIter=maxIter)
                this_model = als.fit(train)
                predict_df = this_model.transform(val)
                
                
                if metric == 'RMSE':   # No need of Top 500
                    
                    # Remove NaN values from prediction  --  Necessary ???
                    #predicted_ratings_df = predict_df.filter(predict_df.prediction != float('nan'))
                    
                    # Round floats to whole numbers  --  TA said no need to round to integer
                    #predicted_ratings_df = predicted_ratings_df.withColumn("prediction", f.abs(f.round(predicted_ratings_df["prediction"],0)))
                          
                    # Evaluate predict_rating_df using chosen evaluator
                    this_score = eval_RMSE.evaluate(predict_df)
                    
                    #scores[i][j][p] = this_score
                    #models[i][j][p] = this_model
                    
                    if this_score < best_score:
                        best_score = this_score
                        best_params = [i, j, p]
                        best_model = this_model
                        
                else:
                    # Ranking Metrics with Top 500 recommendations
                    this_score, other2 = Ranking_evaluator(spark, this_model, val, metric)
                    
                    #scores[i][j][p] = this_score
                    #models[i][j][p] = this_model
                    
                    if this_score > best_score:
                        best_score = this_score
                        best_params = [i, j, p]
                        best_model = this_model
                
                
                # Print current score
                print('For rank %s, regParam %s, maxIter %s : the %s is %s' % (rank, regParam, maxIter, metric ,this_score))
                
                # Print 2 reference ranking scores
                if other2 != None:
                    print('Other Reference Ranking Metrics in this setting:\n')
                    for key, value in other2.items():
                        print(key,': ',value)                

    #best_model = models[best_params[0]][best_params[1]][best_params[2]]
    print('The best model was trained with rank %s, regParam %s and maxIter %s' 
          % (ranks[best_params[0]], regParams[best_params[1]], maxIters[best_params[2]]))
    
    print('The best model has %s of value: %s' % (metric, best_score))
    
    if best_model == None:
        print('Returned model is None. Error???')
        
    # Record End Time & Print Run Time
    fit_end = timeit.default_timer()
    print('Fitting Run Time: ', fit_end - fit_start)
    
    return best_model



#-----------------Recommender & Cold Start comparison----------

def RecSys_ColdStart(spark, train, val, seed = 42,rank = 10, regParam = 0.015, maxIter=10, fraction=0.01, load_path = True):
    
    # Drop a set of book from train (fraction of unique book in val)
    val.createOrReplaceTempView('val')                        
    val_book = spark.sql('SELECT DISTINCT book_id FROM val')
    cold_keep, cold_remove = val_book.randomSplit([1-fraction, fraction], seed=seed)
    new_train= train.join(cold_remove, on=['book_id'], how='left_anti')
    
    # Build the recommendation model using ALS on the training data
    #if load_path == True:
        #cold_model = ALSModel.load('hdfs:/user/yw2115/cold_model_f001_r10')
    #else:

    als = ALS(userCol="user_id", itemCol="book_id", ratingCol="rating",
              coldStartStrategy="nan", implicitPrefs=False, seed = seed)
    als.setParams(rank=rank, regParam=regParam, maxIter=maxIter)
    
    cold_model=als.fit(new_train)
    #cold_model.save('cold_model_f001_r10')

    cold_predict = cold_model.transform(val)
    
    # Get df of user_id, book_id, rating, prediction == NaN (train's unseen book)
    cold_nan=cold_predict.filter(cold_predict.prediction == float('nan'))
    
    # Get df of usesr_id, book_id, rating, prediction != NaN (train's seen book)
    als_predict = cold_predict.filter(cold_predict.prediction != float('nan'))

    #Using functions from coldstart.py
    #load the book attribute matrix
    #if load_path == True: 
        #directly load the transformed matrix
        #book_at = spark.read.parquet('hdfs:/user/yw2115/book_at_100.parquet')
    #else: 
        #build matrix from scratch using the 3 supplement datase
    book_at = build_attribute_matrix(spark, sub = 0.01, book_df='hdfs:/user/yw2115/goodreads_books.json.gz',author_df='hdfs:/user/yw2115/goodreads_book_authors.json.gz',genre_df='hdfs:/user/yw2115/gooreads_book_genres_initial.json.gz',records_path="hdfs:/user/xc1511/onepct_int_001.parquet")
    book_at = k_means_transform(book_at,k=1000,load_model = True)

    #load the book(item) and user latent factor matrix from cold_model
    latent_matrix,user_latent = load_latent(cold_model)
    #user_latent cols: user_id, features

    #get the cold-start book ids
    book_lst = [int(x.book_id) for x in cold_remove.select('book_id').collect()]
    #predict the latent factor for the cold-start books
    cold_pred = []
    for book_id in book_lst:
        print('id',book_id)
        pred = attribute_to_latent_mapping(spark,book_id,book_at,latent_matrix,10,all_data = False)
        print('pred',pred)
        cold_pred.append(pred)
    cold_pred_df = sqlContext.createDataFrame(zip(book_lst, cold_pred), schema=['book_id', 'pred_latent'])
    cold_pred_df = cold_pred_df.filter(cold_pred_df.pred_latent!=''nan'')

   #join the 3 df: predicted latent factor for cold-start books; latent factor for users; cold_nan 
    user_latent.createOrReplaceTempView('user_latent')
    cold_pred_df.createOrReplaceTempView('cold_pred_df')
    cold_nan.createOrReplaceTempView('cold_nan')

    cold_prediction = spark.sql('SELECT cold_nan.user_id, cold_nan.book_id, cold_nan.rating,\
        cold_pred_df.pred_latent AS book_latent, user_latent.features AS user_latent FROM \
        cold_pred_df JOIN cold_nan ON cold_pred_df.book_id = cold_nan.book_id \
        JOIN user_latent ON cold_nan.user_id = user_latent.user_id')

    def dot_p(f1,f2):
        return float(f1.dot(f2))

    dot_product = f.udf(dot_p, FloatType())
    cold_prediction = cold_prediction.withColumn('prediction',dot_product('book_latent','user_latent'))
    cold_prediction = cold_prediction.drop('book_latent').drop('user_latent')

    
    # NEXT STEP ????
    # 1. apply cold_start on cold_nan
    # 2. dot product on user matrix  -- confirmed with TA, right way!!!
    # 3. fill NaN with cold_start prediction
    # 4. Output: cold_predict able to union with als_predict

    #>>> cold_nan.printSchema()
    #root
    #|-- user_id: integer (nullable = true)
    #|-- book_id: integer (nullable = true)
    #|-- rating: float (nullable = true)
    #|-- prediction: float (nullable = False)
    
    
    # Merge als prediction & cold start prediction for evaluation
    merge_predict = als_predict.union(cold_prediction)
    
    # Generate RMSE score for merged prediction df : user_id, book_id, rating, prediction (no NaN this time)
    eval_RMSE = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    merge_score = eval_RMSE.evaluate(merge_predict)
       
    print('Stimulate cold-start by hold out fraction = %s of book in val during training' % (fraction))
    print('For rank %s, regParam %s, maxIter %s : the RMSE is %s' % (rank, regParam, maxIter, merge_score))


#-----------------Recommender Test-----------------------------


def RecSys_test(spark, test, best_model):
    
    # Record Start Time
    test_start=timeit.default_timer()
    
    predict_test_df = best_model.transform(test)  
    
    # Generate RMSE
    predict_test_df = predict_test_df.filter(predict_test_df.prediction != float('nan'))    
    eval_RMSE = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")   
    test_RMSE = eval_RMSE.evaluate(predict_test_df)
    print('RMSE of Best Model on Test Set: ', test_RMSE,'\n')
    
    # Generate Precision at 500
    test_precision,_ = Ranking_evaluator(spark, best_model, test, 'Precision')
    print('Precition at 500 of Best Model on Test Set: ', test_precision ,'\n')
    
    # Generate MAP
    test_MAP,_ = Ranking_evaluator(spark, best_model, test, 'MAP')
    print('MAP of Best Model on Test Set: ', test_MAP,'\n')
    
    # Generate NDCG at 500
    test_NDCG,_ = Ranking_evaluator(spark,best_model, test, 'NDCG')
    print('NDCG at 500 of Best Model on Test Set: ', test_NDCG,'\n')

    # Record End Time
    test_end=timeit.default_timer()
    print('Testing Run Time:', test_end - test_start)
    
    

#-----------------Ranking Evaluator----------------------------

     
def Ranking_evaluator (spark,model, val, metric_type):
    
    val.createOrReplaceTempView('val')                        
    val_user = spark.sql('SELECT DISTINCT user_id FROM val')  
    #val_user = val.select('user_id').distinct()
    val_rec = model.recommendForUserSubset(val_user,500)
    #val_rec.printSchema()
    
    val_rec = val_rec.select('user_id','recommendations',f.posexplode('recommendations')).drop('pos').drop('recommendations')
    val_rec = val_rec.select('user_id',f.expr('col.book_id'),f.expr('col.rating'))
    
    w= Window.partitionBy('user_id')
    val_recrank=val_rec.select('user_id',f.collect_list('book_id').over(w).alias('rec_rank')).sort('user_id').distinct()
   
    val = val.sort(f.desc('rating'))
    val_truerank=val.select('user_id', f.collect_list('book_id').over(w).alias('true_rank')).sort('user_id').distinct()
    
    scoreAndLabels = val_recrank.join(val_truerank,on=['user_id'],how='inner')
    
    rankLists=scoreAndLabels.select("rec_rank", "true_rank").rdd.map(lambda x: tuple([x[0],x[1]])).collect()
    ranks = spark.sparkContext.parallelize(rankLists)
    
    metrics = RankingMetrics(ranks)
    
    MAP = metrics.meanAveragePrecision
    Precision = metrics.precisionAt(500)
    NDCG = metrics.ndcgAt(500)
    
    if metric_type == 'Precision':
        return Precision, {'MAP': MAP,'NDCG': NDCG}
    elif metric_type == 'MAP':
        return MAP, {'Precision': Precision,'NDCG': NDCG}
    elif metric_type == 'NDCG':
        return NDCG, {'MAP': MAP, 'Precision': Precision}
    else:
        return None




