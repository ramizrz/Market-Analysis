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
X=dataset.iloc[:,[0,17]].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X =LabelEncoder()
X[:,1]=labelencoder_X.fit_transform(X[:,1])

def elbow():
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)


def visual():
    # Using the elbow method to find the optimal number of clusters
    elbow()
    # Fitting K-Means to the dataset
    kmeans = KMeans(n_clusters = 13, init = 'k-means++', random_state = 0)
    y_kmeans = kmeans.fit_predict(X)

    # Visualising the clusters
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, cmap = 'FFFF00', label = 'NYC')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, cmap = 'CCFF33', label = 'Reims')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, cmap = '99CCFF', label = 'Paris')
    plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, cmap = 'CCCC66', label = 'pasadena')
    plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, cmap = 'FF99FF', label = 'San Francisco')
    plt.scatter(X[y_kmeans == 5, 0], X[y_kmeans == 5, 1], s = 100, cmap = 'FF9999', label = 'Burlingame')
    plt.scatter(X[y_kmeans == 6, 0], X[y_kmeans == 6, 1], s = 100, cmap = '99CC00', label = 'Lille')
    plt.scatter(X[y_kmeans == 7, 0], X[y_kmeans == 7, 1], s = 100, cmap = 'FF9999', label = 'Bergen')
    plt.scatter(X[y_kmeans == 8, 0], X[y_kmeans == 8, 1], s = 100, cmap = 'FF9966', label = 'Melbourne')
    plt.scatter(X[y_kmeans == 9, 0], X[y_kmeans == 9, 1], s = 100, cmap = 'CC99FF', label = 'Newark')
    plt.scatter(X[y_kmeans == 10, 0], X[y_kmeans == 10, 1], s = 100, cmap = '33CC99', label = 'Bridgewater')
    plt.scatter(X[y_kmeans == 11, 0], X[y_kmeans == 11, 1], s = 100, cmap = 'CC99CC', label = 'Nantes')
    plt.scatter(X[y_kmeans == 12, 0], X[y_kmeans == 12, 1], s = 100, cmap = 'CC9999', label = 'Cambridge')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
    plt.title('Qualitative Analysis')
    plt.xlabel('Sales')
    plt.ylabel('City')
    plt.legend()
    plt.show()

def target():

    kmeans = KMeans(n_clusters = 13, init = 'k-means++', random_state = 0)
    y_kmeans = kmeans.fit_predict(X)

    from bokeh.plotting import figure, output_file, show
    p=figure(plot_width=500,plot_height=400)

    df=pd.read_csv('sales_data_sample.csv',parse_dates=["ORDERDATE"])
    p.xaxis.axis_label = "City"
    p.yaxis.axis_label = "Sales"
    p.vbar(x=X[y_kmeans == 0, 0], top=X[y_kmeans == 0, 1],width=8,color="red",legend='NYC')
    p.vbar(x=X[y_kmeans == 1, 0], top=X[y_kmeans == 1, 1],width=5,color="grey",legend='Reims')
    p.vbar(x=X[y_kmeans == 2, 0], top=X[y_kmeans == 2, 1],width=8,color="green",legend='paris')
    p.vbar(x=X[y_kmeans == 3, 0], top=X[y_kmeans == 3, 1],width=5,color="black",legend='pasadena')
    p.vbar(x=X[y_kmeans == 4, 0], top=X[y_kmeans == 4, 1],width=8,color="yellow",legend='San Francisco')
    p.vbar(x=X[y_kmeans == 7, 0], top=X[y_kmeans == 7, 1],width=10,color="blue",legend='Bergen')
    output_file("result.html")
    show(p)

visual()
target()
