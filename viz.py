# Extension 1: 
# Exploration: use the learned representation to develop a visualization of the items and users, 
# e.g., using T-SNE or UMAP. The visualization should somehow integrate additional information 
# (features, metadata, or genre tags) to illustrate how items are distributed in the learned space.

# References: 
# https://www.liip.ch/en/blog/the-magic-of-tsne-for-visualizing-your-data-features
# https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
# https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1
# https://github.com/DmitryUlyanov/Multicore-TSNE

def viz_rep(model, json_path, item = True, samp_num = 1000):

    """
    required: add the json file to hdfs using command "scp Users/lisacombs/Documents/BIGDATA/goodreads_book_genres_initial.json eac721@dumbo.hpc.nyu.edu:eac721"
    
    model: the best model output from the recommender system fitting and finetuning
    json_path: path to the genres dataset to match book_id with genre
    item: use book latent factors

    """

    #dir(model)
    #model.itemFactors.show()
    #model.userFactors.show()

    i = model.itemFactors.toPandas()
    i2 = i.features.apply(pd.Series)
    i2['item_id'] = i['id']

    items = i.sample(n=10000, random_state=1)

    # merge with genres
    genres=spark.read.json("genres.json", multiLine=True)
    X=items.join(genres, on='book_id', how = 'inner')

    #items.to_csv(index = False)
    #X = pd.read_csv('items.csv')

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    embeddings = tsne.fit_transform(X.iloc[:,-1])
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]

    #./bin/spark-submit mypythonfile.py

    return vis_x, vis_y, X.genre

def tsneplot(vis_x, vis_y, X.genre):
    import matplotlib
    matplotlib.use('Agg')

    matplotlib.pyplot.scatter(vis_x, vis_y, c=X.genre, cmap=plt.cm.get_cmap("jet", 10), marker='.')
    matplotlib.pyplot.colorbar(ticks=range(10))
    matplotlib.pyplot.clim(-0.5, 9.5)
    matplotlib.pyplot.show()

    plt.savefig('tsne_test.png')