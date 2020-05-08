#!/usr/bin/env python

# rec_fit & rec_test & ranking_evaluator

from pyspark.ml.recommendation import ALS #, Rating
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as f
import timeit
from pyspark.sql import Window
from pyspark.mllib.evaluation import RankingMetrics



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
