# rec_fit & rec_test & ranking_evaluator

from pyspark.ml.recommendation import ALS #, Rating
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as f
from pyspark.sql.types import DoubleType
import numpy as np



#----------------Recommender Fit------------------------

def RecSys_fit (train, val, metric = 'RMSE', seed = 42,ranks = [10, 15], 
            regParams = [0.01, 0.15], maxIters = [10]):
    

    # Build the recommendation model using ALS on the training data
    als = ALS(userCol="user_id", itemCol="book_id", ratingCol="rating",
              coldStartStrategy="drop", implicitPrefs=False, seed = seed)
    
    if metric == 'RMSE':
        eval_RMSE = RegressionEvaluator(metricName="rmse", labelCol="rating", 
                                        predictionCol="prediction")
        # Initialize best_score to track min
        best_score = float('inf')
    
    elif metric in ['Precision','MAP','NDCG']:   # Are we doing MAP or other ranking metric???
        
        #Initialize best_score to track max
        best_score = -1*float('inf')  
    
    else:
        raise ValueError("Score metric not supported.")
    
    models = np.zeros([len(ranks), len(regParams), len(maxIters)])
    scores = np.zeros([len(ranks), len(regParams), len(maxIters)])
    
    
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
                    predicted_ratings_df = predict_df.filter(predict_df.prediction != float('nan'))
                    
                    # Round floats to whole numbers  --  TA said no need to round to integer
                    #predicted_ratings_df = predicted_ratings_df.withColumn("prediction", f.abs(f.round(predicted_ratings_df["prediction"],0)))
                          
                    # Evaluate predict_rating_df using chosen evaluator
                    this_score = eval_RMSE.evaluate(predicted_ratings_df)
                    
                    scores[i][j][p] = this_score
                    models[i][j][p] = this_model
                    
                    if this_score < best_score:
                        best_score = this_score
                        best_params = [i, j, p]
                        
                else:
                    # Ranking Metrics with Top 500 recommendations
                    this_score = Ranking_evaluator(this_model, val, metric)
                    
                    scores[i][j][p] = this_score
                    models[i][j][p] = this_model
                    
                    if this_score > best_score:
                        best_score = this_score
                        best_params = [i, j, p]
                
                
                # Print current score
                print('For rank %s, regParam %s, maxIter %s : the %s is %s' % (rank, regParam, maxIter, metric ,this_score))
                
                
    best_model = models[best_params[0]][best_params[1]][best_params[2]]
    print('The best model was trained with rank %s, regParam %s and maxIter %s' 
          % (ranks[best_params[0]], regParams[best_params[1]], maxIters[best_params[2]]))
    
    print('The best model has %s of value %s:' % (metric, best_score))
                
    return best_model
        



#-----------------Recommender Test-----------------------------


def RecSys_test(test, best_model):
    
    # TO BE CONTINUE
    
    predict_test_df = best_model.transform(test)
    
    
    # Generate RMSE
    predict_test_df = predict_test_df.filter(predict_test_df.prediction != float('nan'))    
    eval_RMSE = RegressionEvaluator(metricName="rmse", labelCol="rating", 
                                    predictionCol="prediction")   
    test_RMSE = eval_RMSE.evaluate(predict_test_df)
    print('RMSE of Best Model on Test Set: ', test_RMSE,'\n')
    
    # Generate Precision at 500
    test_precision = Ranking_evaluator(best_model, test, 'Precision')
    print('Precition at 500 of Best Model on Test Set: ', test_precision)
    
    # Generate MAP
    test_MAP = Ranking_evaluator(best_model, test, 'MAP')
    print('MAP of Best Model on Test Set: ', test_MAP)
    
    # Generate NDCG at 500
    test_NDCG = Ranking_evaluator(best_model, test, 'NDCG')
    print('NDCG at 500 of Best Model on Test Set: ', test_NDCG)

    
    


#-----------------Ranking Evaluator----------------------------

from pyspark.sql.functions import *
from pyspark.sql import Window
from pyspark.mllib.evaluation import RankingMetrics
    
    
    
def Ranking_evaluator (model, val, metric_type):
    
    
    val.createOrReplaceTempView('val')
    val_user = spark.sql('SELECT DISTINCT user_id FROM val')
    val_rec = model.recommendForUserSubset(val_user,500)
    #val_rec.printSchema()
    
    val_rec = val_rec.select('user_id','recommendations',posexplode('recommendations')).drop('pos').drop('recommendations')
    val_rec = val_rec.select('user_id',f.expr('col.book_id'),f.expr('col.rating'))
    
    w= Window.partitionBy('user_id')
    val_recrank=val_rec.select('user_id',collect_list('book_id').over(w).alias('rec_rank')).sort('user_id').distinct()
   
    val = val.sort(desc('rating'))
    val_truerank=val.select('user_id', collect_list('book_id').over(w).alias('true_rank')).sort('user_id').distinct()
    
    scoreAndLabels = val_recrank.join(val_truerank,on=['user_id'],how='inner')
    
    rankLists=scoreAndLabels.select("rec_rank", "true_rank").rdd.map(lambda x: tuple([x[0],x[1]])).collect()
    ranks = spark.sparkContext.parallelize(rankLists)
    
    metrics = RankingMetrics(ranks)
    
    #print(metrics.meanAveragePrecision)
    #print(metrics.precisionAt(500))
    #print(metrics.ndcgAt(500))
    
    if metric_type == 'Precision':
        return metrics.precisionAt(500)
    elif metric_type == 'MAP':
        return metrics.meanAveragePrecision
    elif metric_type == 'NDCG':
        return metrics.ndcgAt(500)
    else:
        return None



