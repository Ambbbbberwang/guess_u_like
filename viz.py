# Extension 1: 
# Exploration: use the learned representation to develop a visualization of the items and users, 
# e.g., using T-SNE or UMAP. The visualization should somehow integrate additional information 
# (features, metadata, or genre tags) to illustrate how items are distributed in the learned space.

# References: 
# https://www.liip.ch/en/blog/the-magic-of-tsne-for-visualizing-your-data-features
# https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
# https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1
# https://github.com/DmitryUlyanov/Multicore-TSNE

# pyspark
def viz_rep(model, item = True, samp_num = 1000, seed = 42):

    """
    required: 
    
    model: the best model output from the recommender system fitting and finetuning
    json_path: path to the genres dataset to match book_id with genre
    item: use book latent factors

    """

    import pandas as pd 

    #dir(model)
    #model.itemFactors.show()
    #model.userFactors.show()

    i = model.itemFactors
    i.createOrReplaceTempView('i')
    items = spark.sql('SELECT DISTINCT id FROM i')
    samp_pct = samp_num/items.count()
    graph_i, not_graph_i = items.randomSplit([samp_pct, 1-samp_pct], seed=seed)

    graph_i=graph_i.join(i, on='id', how='inner').toPandas()
    i2 = graph_i.features.apply(pd.Series)
    i2['item_id'] = graph_i['id']
    
    i2.to_csv('items_test.csv')

# python
def tsneplot(items_path='items_test.csv', genres_path='goodreads_book_genres_initial.json', fig_path = 'tsne.png'):

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE

    genres = []
    for line in open(genres_path, 'r'):
        genres.append(json.loads(line))

    df_genres = pd.DataFrame(genres)

    genres2 = []
    for i in range(0, len(df_genres['genres'])):
        if len(df_genres['genres'][i]) > 0:
            genres2.append(list(df_genres['genres'][i].keys())[0])
        else: 
            genres2.append(None)

    df_genres['top-genre'] = genres2
    df_genres=df.astype({'book_id': 'int'})

    items = pd.read_csv(items_path)
    items = items.drop("Unnamed: 0", axis=1)
    items=items.rename(columns = {'item_id': 'book_id'})

    merged_items = pd.merge(items, df_genres, on='book_id', how = 'inner')
    merged_items=merged_items.drop('genres', axis=1)

    tsne = TSNE(n_components=2, random_state=seed)
    tsne_obj= tsne.fit_transform(merged_items.iloc[:,0:merged_items.shape[1]-1])

    tsne_df = pd.DataFrame({'X':tsne_obj[:,0],'Y':tsne_obj[:,1],'top-genre':merged.iloc[:,-1]})

    sns_plot = sns.scatterplot(x="X", y="Y",hue="top-genre", palette=sns.color_palette("muted"),legend='full', data=tsne_df)

    sns_plot.savefig('tsne_test.png')