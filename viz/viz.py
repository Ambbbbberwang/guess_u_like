#!/usr/bin/env python

# Extension 1: 
# Exploration: use the learned representation to develop a visualization of the items and users, 
# e.g., using T-SNE or UMAP. The visualization should somehow integrate additional information 
# (features, metadata, or genre tags) to illustrate how items are distributed in the learned space.

# References: 
# https://www.liip.ch/en/blog/the-magic-of-tsne-for-visualizing-your-data-features
# https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
# https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1

# run using: 
# python viz.py

def tsneplot(points = 10000, seed = 42, fig_path = 'tsne.png'):

    """
    points: number of points to visualizse
    seed = random_state
    fig_path: where to save the plot

    return None, saves plot

    """
    
    import matplotlib
    matplotlib.use('Agg')

    import pandas as pd
    from glob import glob
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import TSNE
    from sklearn import preprocessing

    plt.rcParams['figure.figsize'] = [20,20]
    
    filename = glob("*.csv")[0]
    print(filename)
    items = pd.read_csv(filename, engine='python', header=None)
    size=items.shape[0]
    print('read data.')

    # subset data
    items = items.iloc[0:points, :]
    num_features = items.shape[1]
    # get the same visualization order each time
    items=items.sort_values(items.columns[-1])

    # standardize the latent factors
    X = items.iloc[:,1:num_features-1]
    standardized_X = preprocessing.scale(X)
    print('processed data.')

    # compute tsne : using complexity rules - towardsdatascience.com/how-to-tune-hyperparameters-of-tsne-7c0596a18868
    tsne = TSNE(n_components=2, random_state=seed, perplexity=size**0.5, n_iter = 750)
    tsne_obj= tsne.fit_transform(items.iloc[:,1:num_features-1])
    tsne_df = pd.DataFrame({'X: tSNE Component 1':tsne_obj[:,0],'Y: tSNE Component 2':tsne_obj[:,1],'top-genre':items.iloc[:,-1]})
    print('tsne data.')

    #print(tsne_df)
    
    # plot data
    sns.set(font_scale=2.25) 
    sns.set_style("white")

    sns_plot = sns.scatterplot(x="X: tSNE Component 1", y="Y: tSNE Component 2", hue="top-genre", palette=sns.color_palette("Paired", 10),legend='full', data=tsne_df)
    sns_plot.legend(loc=2)
    sns_plot.set(ylim=(-30, 30))
    sns_plot.set(xlim=(-30, 30))
    sns_plot.set_title('tSNE Dimesionality Reduction of Item Factors from ALS Model by Genre')
    print('plotted data.')

    sns_plot.figure.savefig(fig_path)

# run to build plot
tsneplot()
