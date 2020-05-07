#!/usr/bin/env python

# Extension 1: 
# Exploration: use the learned representation to develop a visualization of the items and users, 
# e.g., using T-SNE or UMAP. The visualization should somehow integrate additional information 
# (features, metadata, or genre tags) to illustrate how items are distributed in the learned space.

# References: 
# https://www.liip.ch/en/blog/the-magic-of-tsne-for-visualizing-your-data-features
# https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
# https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1
# https://github.com/DmitryUlyanov/Multicore-TSNE

# python tsneplot()

def tsneplot(items_path='hdfs:/user/eac721/items.csv', rank = 15, fig_path = 'tsne.png'):

    """
    items_path='items_matrix.csv' : load the matrix with latent factors, id, genre
    rank: how many features are there?
    fig_path: where to save the plot

    return None, saves plot

    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
    import pandas as pd

    items = pd.read_csv(items_path)

    tsne = TSNE(n_components=2, random_state=seed)
    tsne_obj= tsne.fit_transform(items.iloc[:,1:rank])
    tsne_df = pd.DataFrame({'X':tsne_obj[:,0],'Y':tsne_obj[:,1],'top-genre':items.loc[:,'']})

    sns_plot = sns.scatterplot(x="X", y="Y", hue="top-genre", palette=sns.color_palette("muted"),legend='full', data=tsne_df)
    sns_plot.savefig('tsne_test.png')

tsneplot(fig_path = 'tsne.png')