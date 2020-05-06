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

book_path = 'hdfs:/user/yw2115/goodreads_books.json.gz'
author_path = 'hdfs:/user/yw2115/goodreads_book_authors.json.gz'
genre_path = 'hdfs:/user/yw2115/gooreads_book_genres_initial.json.gz'

def load_data(file_name, head = 500):
    count = 0
    data = []
    with gzip.open(file_name) as fin:
        for l in fin:
            d = json.loads(l)
            count += 1
            data.append(d)
            
            # break if reaches the 100th line
            if (head is not None) and (count > head):
                break
    #return spark dataframe
    data_df = spark.createDataFrame(data)

    return data_df

book_df = load_data(book_path, head = 500)
author_df = load_data(author_path, head = 500)
genre_df = load_data(genre_path, head = 500)



