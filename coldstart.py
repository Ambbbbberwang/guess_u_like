#!/usr/bin/env python

'''

Extension 2: Cold-start
Using the supplementary book data, build a model that can map observable data
to the learned latent factor representation for items. To evaluate its 
accuracy, simulate a cold-start scenario by holding out a subset of 
items during training (of the recommender model), and compare its performance 
to a full collaborative filter model.

'''

#Reference: https://github.com/MengtingWan/goodreads/blob/master/samples.ipynb
import pyspark.sql.functions as f
from pyspark.sql.functions import when
from pyspark.ml.feature import VectorAssembler


####Load Supplement Book Data####

def build_attribute_matrix(spark, sub = 0, book_df='hdfs:/user/yw2115/goodreads_books.json.gz',author_df='hdfs:/user/yw2115/goodreads_book_authors.json.gz',genre_df='hdfs:/user/yw2115/gooreads_book_genres_initial.json.gz',records_path="hdfs:/user/xc1511/onepct_int_001.parquet"):

    ####Create Attribute Matrix for Genres####
    '''
    10 categories: 
    children| comics, graphic| fantasy, paranormal| fiction| history, historical fiction, biography,
    mystery, thriller, crime| non-fiction| poetry| romance| young-adult

    '''

    book_df = spark.read.json('hdfs:/user/yw2115/goodreads_books.json.gz')
    author_df =spark.read.json('hdfs:/user/yw2115/goodreads_book_authors.json.gz')
    genre_df =spark.read.json('hdfs:/user/yw2115/gooreads_book_genres_initial.json.gz')

    genre_at = genre_df.select('book_id',f.expr('genres.children'),f.expr('genres.`comics, graphic`'),\
        f.expr('genres.`fantasy, paranormal`'),f.expr('genres.fiction'), \
        f.expr('genres.`history, historical fiction, biography`'), f.expr('genres.`mystery, thriller, crime`'),\
        f.expr('genres.`non-fiction`'),f.expr('genres.poetry'),f.expr('genres.romance'),f.expr('genres.`young-adult`'))

    #change col names
    new_col = ['book_id','g1','g2','g3','g4','g5','g6','g7','g8','g9','g10']
    genre_at = genre_at.toDF(*new_col)
    #genre_at.show(3)

    #0/1 Encoding
    #change Null value to 0 (meaning the book is not in this genre) 
    # and other int to 1 (meaning the book in this genre)

    for i in range(1,len(new_col)):
        col_name = new_col[i]
        genre_at = genre_at.withColumn(col_name, when(genre_at[col_name].isNotNull(), 1).otherwise(0))

    #genre_at.show(10)

    #subsample 1% data
    if sub == 0.01:
        records_pq = spark.read.parquet(records_path)
        records_pq.createOrReplaceTempView('records_pq')
        book_pq = spark.sql('SELECT DISTINCT book_id FROM records_pq')

        book_pq.createOrReplaceTempView('book_pq')
        book_df.createOrReplaceTempView('book_df')
        genre_at.createOrReplaceTempView('genre_at')
        genre_at = spark.sql('SELECT genre_at.* FROM genre_at JOIN book_pq ON \
            genre_at.book_id = book_pq.book_id')
        book_df = spark.sql('SELECT book_df.* FROM book_df JOIN book_pq ON \
            book_df.book_id = book_pq.book_id')

    ####Add Author Rating as Additional Attribute####
    #Select the first author (there are books with more than 1 author, first author is the main author)
    book_df = book_df.select('book_id',f.expr('authors[0]').alias('a'))
    #Add author_id
    book_df = book_df.select('book_id',f.expr('a.author_id'))


    #Join book_df and author_df
    book_df.createOrReplaceTempView('book_df')
    author_df.createOrReplaceTempView('author_df')

    author_at = spark.sql('SELECT book_df.book_id, book_df.author_id,\
     author_df.average_rating FROM book_df JOIN author_df ON \
     book_df.author_id=author_df.author_id')
    #author_at.show(10)

    ####Join The Two Matrix to Get Book Attribute Matrix####
    genre_at.createOrReplaceTempView('genre_at')
    author_at.createOrReplaceTempView('author_at')

    book_at = spark.sql('SELECT genre_at.book_id, genre_at.g1, genre_at.g2,\
     genre_at.g3, genre_at.g4, genre_at.g5, genre_at.g6, genre_at.g7, genre_at.g8, \
     genre_at.g9, genre_at.g10, author_at.average_rating AS author_rating \
     FROM genre_at JOIN author_at ON genre_at.book_id=author_at.book_id')

    book_at = book_at.withColumn('author_rating',book_at['author_rating'].cast('float'))

    #return the I*N attribute matrix for book
    #I is number of items (books)
    #N = 11 is number of attribute features of the books

    #add a features col
    vecAssembler = VectorAssembler(inputCols=['g1','g2','g3','g4','g5','g6','g7','g8','g9','g10','author_rating'], outputCol="features")
    book_at = vecAssembler.transform(book_at)
    #note here 'features' is a SparseVector type due to spark memory default

    #book_at.show(3)

    return book_at



####Load latent factor for books and users####
def load_latent(model):

    #from pyspark.sql.functions import *
    #from pyspark.sql import selectExpr
    
    latent = model.itemFactors 
    user = model.userFactors
    #a DataFrame that stores item factors in two columns: id and features
    size = model.rank
    col = ['id']
    for x in range(size):
        col.append('features[' + str(x) + ']')

    #Convert to latent matrix with first col as "book_id" the rest col are latent features
    latent = latent.selectExpr(col)
    user = user.selectExpr(col)
    #rename cols
    new_col = ['book_id']
    user_col = ['user_id']
    f_col = []
    for i in range(size):
        new_col.append('f'+str(i))
        f_col.append('f'+str(i))
        user_col.append('f'+str(i))
    #return latent matrix
    latent_matrix = latent.toDF(*new_col)
    user_latent = user.toDF(*user_col)
    #load user latent factor 
    vecAssembler = VectorAssembler(inputCols=f_col, outputCol="features")
    user_latent = vecAssembler.transform(user_latent)
    user_latent = user_latent.select('user_id','features')


    return latent_matrix,user_latent



####Attribute-to-Latent_Factor Mapping####
###k_means clustering for faster knn calculation###
def k_means_transform(book_at,k=100,load_model = True):
    '''
    input: attribute feature matrix of all books
    output: transformed matrix including cluster assignment
    This function is used to cluster all books for faster calculation for knn later
    '''

    if load_model == False:

        ###k-means clustering###
        #Since the data is too big to do knn, first cluster them
        from pyspark.ml.clustering import KMeans
        kmeans = KMeans(k=k, seed=42) #divide all books to 1000 clusters (1/1000, less computation for knn)
        model = kmeans.fit(book_at.select('features'))
        #model.save('k-means_model_001_10')
    else:
        from pyspark.ml.clustering import KMeansModel
        model = KMeansModel.load('hdfs:/user/yw2115/k-means_model_001')

    #add the cluster col to original attribute matrix
    transformed = model.transform(book_at)
    transformed = transformed.withColumnRenamed("prediction","cluster")
    #transormed.show(3)
    return transformed

###Compute cosine similarity between two columns###
def cos_sim(f1,f2):
    return float(f1.dot(f2) / (f1.norm(2) * f2.norm(2))) 



###k-Nearest-Neighbors Mapping###

def get_k_nearest_neighbors(spark,book_id,book_at,k):
    '''
    input: book_id for the cold start book
    book_at: should be the transformed attribute matrix with cluster assignment (for faster calculation)
    k: number of nearest neighbors
    '''
    from pyspark.sql import functions as f
    from pyspark.sql.types import FloatType

    if book_at.where(book_at.book_id == book_id).rdd.isEmpty() == True:
        return 0,0
    else:

        #load the dataframe with a single row of target book
        target_book = book_at.where(book_at.book_id == book_id)
        #target_book.show()
        target_book.createOrReplaceTempView('target_book')
        book_at.createOrReplaceTempView('book_at')

        
        ###get the sub_data of the same cluster as the target###
        sub_data = spark.sql('SELECT target_book.book_id AS target_id, target_book.features AS features1, target_book.cluster, \
            book_at.book_id, book_at.features AS features2 FROM target_book JOIN book_at ON \
            target_book.cluster = book_at.cluster')
        #sub_data.show(10)
        #DataFrame[target_id: string, features1: vector, cluster: int, book_id: string, features2: vector]

        ###calculate the cosine similarity###
        def cos_sim(f1,f2):
            return float(f1.dot(f2) / (f1.norm(2) * f2.norm(2))) 

        cosine_similarity = f.udf(cos_sim, FloatType())
        sub_data = sub_data.withColumn('cosine_similarity',cosine_similarity('features1','features2'))

        ###get k nearest neighbors with highest cosine similarity###
        knn_df = sub_data.select('book_id','cosine_similarity').sort('cosine_similarity',ascending=False).limit(k)
        cluster_df = sub_data.select('book_id','cosine_similarity') #all books in the same cluster

        return knn_df, cluster_df





def attribute_to_latent_mapping(spark,book_id,book_at,latent_matrix,k,all_data = False):
    '''
    input: 
    book_id: the book id of cold start item
    book_at: attribute_matrix of size (I*N) -- transformed book_at with cluster assignment
    latent_matrix of size (I*K)
    k: k nearest neighbors for mapping
    I: number of total items (books) 
    N: number of observable content features of a book
    K: rank in the model, also number of latent factors
    all_data: if all_data == False, it means the latent matrix extracted from the model is only trained
    on 1%, 5% or 25% data, knn of the target book may not included in the training set. In this case, we
    use average latent factor of books in the same cluster rather than weighted knn as prediction. 
    Note: the cosine_similarities of books in the same cluster are mostly over 0.99. Taking average is the same as weighted average.
    '''
    latent_matrix.createOrReplaceTempView('latent_matrix')
    knn_df, cluster_df = get_k_nearest_neighbors(spark, book_id,book_at,k) #cols: book_id, cosine similarity

    if cluster_df == 0:
        return 'nan'
    else:
        if all_data == True:
            knn_df.createOrReplaceTempView('knn_df')
            map_latent = spark.sql('SELECT latent_matrix.*, knn_df.cosine_similarity FROM latent_matrix JOIN\
                knn_df ON knn_df.book_id = latent_matrix.book_id')
            
        else: 
            cluster_df.createOrReplaceTempView('cluster_df')
            map_latent = spark.sql('SELECT latent_matrix.*, cluster_df.cosine_similarity FROM latent_matrix JOIN\
                cluster_df ON cluster_df.book_id = latent_matrix.book_id')

        map_latent = map_latent.drop('book_id').drop('cosine_similarity')
        pred_df = map_latent.select(*[f.mean(c).alias(c) for c in map_latent.columns])

        vecAssembler = VectorAssembler(inputCols=pred_df.columns, outputCol="features")
        pred_df = vecAssembler.transform(pred_df)
        pred = pred_df.select('features').collect()[0].features

        return pred





'''
def test_main():
    from pyspark.ml.recommendation import ALS, ALSModel
    #book_at = build_attribute_matrix(spark, book_df='hdfs:/user/yw2115/goodreads_books.json.gz',author_df='hdfs:/user/yw2115/goodreads_book_authors.json.gz',genre_df='hdfs:/user/yw2115/gooreads_book_genres_initial.json.gz')
    book_at = build_attribute_matrix(spark, sub = 0.01, book_df='hdfs:/user/yw2115/goodreads_books.json.gz',author_df='hdfs:/user/yw2115/goodreads_book_authors.json.gz',genre_df='hdfs:/user/yw2115/gooreads_book_genres_initial.json.gz',records_path="hdfs:/user/xc1511/onepct_int_001.parquet")
    transformed = k_means_transform(book_at,k=1000,load_model = False)
    transformed.write.parquet('hdfs:/user/yw2115/book_at_001.parquet')
    #book_id = 3
    #k = 10
    model = ALSModel.load('hdfs:/user/xc1511/001_r200_re0015_m10')
    latent_matrix = load_latent(model)
    pred_df = attribute_to_latent_mapping(spark,book_id,book_at,latent_matrix,k,all_data = False)

'''



