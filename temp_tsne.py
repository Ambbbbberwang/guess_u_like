####Load latent factor for books####
def load_latent(model):

    #from pyspark.sql.functions import *
    #from pyspark.sql import selectExpr
    
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

def build_tsne_matrix(spark, latent_matrix, genre_df='hdfs:/user/yw2115/gooreads_book_genres_initial.json.gz', save_csv='tsne_matrix.csv'):

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
    import pyspark.sql.functions as f

    genre_df =spark.read.json(genre_df)

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

    tsne_matrix.createOrReplaceTempView('spark_df')
    books = spark.sql('SELECT DISTINCT book_id FROM spark_df')
    splits = books.randomSplit([0.25, 0.75], seed=42)
    book_samp = splits[0]

    # save to csv for py script
    book_samp.coalesce(1).write.csv(save_csv)
