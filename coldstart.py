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
