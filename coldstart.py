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

####Load Supplement Book Data####

book_df = spark.read.json('hdfs:/user/yw2115/goodreads_books.json.gz')
author_df =spark.read.json('hdfs:/user/yw2115/goodreads_book_authors.json.gz')
genre_df =spark.read.json('hdfs:/user/yw2115/gooreads_book_genres_initial.json.gz')


def build_attribute_matrix(book_df,author_df,genre_df):

    ####Create Attribute Matrix for Genres####
    '''
    10 categories: 
    children| comics, graphic| fantasy, paranormal| fiction| history, historical fiction, biography,
    mystery, thriller, crime| non-fiction| poetry| romance| young-adult

    '''
    import pyspark.sql.functions as f
    from pyspark.sql.functions import when

    genre_at = genre_df.select('book_id',f.expr('genres.children'),f.expr('genres.`comics, graphic`'),\
        f.expr('genres.`fantasy, paranormal`'),f.expr('genres.fiction'), \
        f.expr('genres.`history, historical fiction, biography`'), f.expr('genres.`mystery, thriller, crime`'),\
        f.expr('genres.`non-fiction`'),f.expr('genres.poetry'),f.expr('genres.romance'),f.expr('genres.`young-adult`'))

    #change col names
    new_col = ['book_id','g1','g2','g3','g4','g5','g6','g7','g8','g9','g10']
    genre_at = genre_at.toDF(*new_col)

    #0/1 Encoding
    #change Null value to 0 (meaning the book is not in this genre) 
    # and other int to 1 (meaning the book in this genre)

    for i in range(1,len(new_col)):
        col_name = new_col[i]
        genre_at = genre_at.withColumn(col_name, when(genre_at[col_name].isNotNull(), 1).otherwise(0))

    #genre_at.show(10)

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

    #return the I*N attribute matrix for book
    #I is number of items (books)
    #N = 11 is number of attribute features of the books
    return book_at



####Load latent factor for books####
def load_latent(model):

    #from pyspark.sql.functions import *
    from pyspark.sql.functions import selectExpr
    
    latent = model.itemFactors 
    #a DataFrame that stores item factors in two columns: id and features
    size = model.rank
    col = ['id']
    for x in range(size):
        col.append('features[' + str(x) + ']')

    #Convert to latent matrix with first col as "book_id" the rest col are latent features
    latent = latent.selectExpr(col)
    #rename cols
    new_col = ['book_id']
    for i in range(size):
        new_col.append('f'+str(i))
    #return latent matrix
    latent_matrix = latent.toDF(*new_col)


    return latent_matrix



####Attribute-to-Latent_Factor Mapping####

###k-Nearest-Neighbors Mapping
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def get_neighbors(item_row,attribute_matrix,k):
    cs = cosine_similarity(item_row,attribute_matrix)
    






    idx = np.argsort(cs)[::-1]
    k_idx = idx[:k]
    score = []
    for i in k_idx:
        score.append(cs[i])




    return score,k_idx



def attribute_to_latent_mapping(attribute_matrix,latent_matrix):
    '''
    input: 
    attribute_matrix of size (I*N)
    latent_matrix of size (I*k)
    I: number of total items (books) 
    N: number of observable content features of a book
    k: rank in the model, also number of latent factors
    '''

    ####



#### Using supplemental data for TSNE ####

def build_tsne_matrix(genre_df, latent_matrix):

    """
    saves the csv for the tsne plot in viz.py
    # reference: https://stackoverflow.com/questions/46179453/how-to-compute-maximum-per-row-and-return-a-colum-of-max-value-and-another-colu

    genre_df: hdfs:/user/yw2115/gooreads_book_genres_initial.json.gz, downloaded from goodreads online
    latent_matrix: output from load_latent(model)

    return: None
    saves: data structure with bookid, lf's from the model, and genre matched
    """

    from pyspark.sql.types import StringType
    from pyspark.sql.functions import col, greatest, udf, array

    genre_at = genre_df.select('book_id',f.expr('genres.children'),f.expr('genres.`comics, graphic`'),\
        f.expr('genres.`fantasy, paranormal`'),f.expr('genres.fiction'), \
        f.expr('genres.`history, historical fiction, biography`'), f.expr('genres.`mystery, thriller, crime`'),\
        f.expr('genres.`non-fiction`'),f.expr('genres.poetry'),f.expr('genres.romance'),f.expr('genres.`young-adult`'))
    #genre_at = genre_at.toDF()
    #genre_only = genre_at.drop('book_id')

    df1 = genre_at.withColumn("maxValue", greatest(*[col(x) for x in genre_at.columns[1:]]))

    col_arr = df1.columns

    def modify_values(r):
        for i in range(len(r[:-1])):
            if r[i]==r[-1]:
                return col_arr[i]

    modify_values_udf = udf(modify_values, StringType())

    df1 = df1.withColumn("maxColumn", modify_values_udf(array(df1.columns)))
    book_genre = df1.select('book_id', 'maxColumn')

    tsne_matrix=latent_matrix.join(book_genre, on='book_id', how='inner')

    # adding save for tsne
    tsne_matrix.to_csv('tsne_matrix.csv')

    



