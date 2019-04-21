# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.special
from sklearn.cluster import KMeans
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file
# Importing the dataset
dataset = pd.read_csv('sales_data_sample.csv',parse_dates=["ORDERDATE"])
X=dataset.iloc[:,[1,17]].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X =LabelEncoder()
X[:,1]=labelencoder_X.fit_transform(X[:,1])


# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
brc = Birch(n_clusters = 13 ,branching_factor=50,threshold=0.5,compute_labels=True)
brc.fit(X)
y_hc= brc.predict(X)


# Visualising the clusters
def visual():
    plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, cmap = 'FFFF00', label = 'NYC')
    plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, cmap = 'CCFF33', label = 'Reims')
    plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, cmap = '99CCFF', label = 'Paris')
    plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, cmap = 'CCCC66', label = 'pasadena')
    plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, cmap = 'FF99FF', label = 'San Francisco')
    plt.scatter(X[y_hc == 5, 0], X[y_hc == 5, 1], s = 100, cmap = 'FF9999', label = 'Burlingame')
    plt.scatter(X[y_hc == 6, 0], X[y_hc == 6, 1], s = 100, cmap = '99CC00', label = 'Lille')
    plt.scatter(X[y_hc == 7, 0], X[y_hc == 7, 1], s = 100, cmap = 'FF9999', label = 'Bergen')
    plt.scatter(X[y_hc == 8, 0], X[y_hc == 8, 1], s = 100, cmap = 'FF9966', label = 'Melbourne')
    plt.scatter(X[y_hc == 9, 0], X[y_hc == 9, 1], s = 100, cmap = 'CC99FF', label = 'Newark')
    plt.scatter(X[y_hc == 10, 0], X[y_hc == 10, 1], s = 100, cmap = '33CC99', label = 'Bridgewater')
    plt.scatter(X[y_hc == 11, 0], X[y_hc == 11, 1], s = 100, cmap = 'CC99CC', label = 'Nantes')
    plt.scatter(X[y_hc == 12, 0], X[y_hc == 12, 1], s = 100, cmap = 'CC9999', label = 'Cambridge')
    plt.title('Quantitative Analysis')
    plt.xlabel('Quantity')
    plt.ylabel('City')
    plt.legend()
    plt.show()

def target():

    brc = Birch(n_clusters = 13 ,branching_factor=50,threshold=0.5,compute_labels=True)
    brc.fit(X)
    y_hc= brc.predict(X)


    from bokeh.plotting import figure, output_file, show
    p=figure(plot_width=500,plot_height=400)

    df=pd.read_csv('sales_data_sample.csv',parse_dates=["ORDERDATE"])
    p.yaxis.axis_label = "City"
    p.xaxis.axis_label = "Quantity"
    p.circle(X[y_hc == 0, 0], X[y_hc == 0, 1],size=8,color="red",legend='NYC')
    p.circle(X[y_hc == 1, 0], X[y_hc == 1, 1],size=5,color="grey",legend='Reims')
    p.circle(X[y_hc == 2, 0], X[y_hc == 2, 1],size=8,color="green",legend='paris')
    p.circle(X[y_hc == 3, 0], X[y_hc == 3, 1],size=5,color="black",legend='pasadena')
    p.circle(X[y_hc == 4, 0], X[y_hc == 4, 1],size=8,color="yellow",legend='San Francisco')
    p.circle(X[y_hc == 7, 0], X[y_hc == 7, 1],size=10,color="blue",legend='Bergen')
    output_file("result1.html")
    show(p)

visual()
target()
