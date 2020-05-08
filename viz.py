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

def tsneplot(seed = 42, fig_path = 'tsne.png'):
    """
    items_path='items_matrix.csv' : load the matrix with latent factors, id, genre
    rank: how many features are there?
    fig_path: where to save the plot

    return None, saves plot

    """

    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
    import pandas as pd
    from glob import glob

    filename = glob("tsne_matrix.csv/*.csv")[0]
    items = pd.read_csv(filename, engine='python', header=None)
    print('read data.')
    print(items.head())
    num_features = items.shape[1]

    # sample data
    items = items.sample(n=1000, random_state=seed, replace=False)
    print('sampled data.')

    tsne = TSNE(n_components=2, random_state=seed)
    tsne_obj= tsne.fit_transform(items.iloc[:,1:num_features-1])
    tsne_df = pd.DataFrame({'X':tsne_obj[:,0],'Y':tsne_obj[:,1],'top-genre':items.iloc[:,-1]})

    print(tsne_df)
    
    print('plotting data.')
    sns_plot = sns.scatterplot(x="X", y="Y", hue="top-genre", palette=sns.color_palette("muted"),legend='full', data=tsne_df)
    sns_plot.figure.savefig('tsne_test.png')

tsneplot(fig_path = 'tsne.png')
