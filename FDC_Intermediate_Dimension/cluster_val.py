import math
import numpy as np
import matplotlib.pyplot as plt


def intra_cluster_dist(cluster_dataframe_list):
    maxi=0
    for df in cluster_dataframe_list:
        for i in range(len(df)):
            for ii in range(i+1,len(df)):
                distance=math.dist(df[i],df[ii])
                maxi=max(maxi,distance)
        
    return maxi

def inter_cluster_dist(cluster_dataframe_list):
    mini=math.inf
    for i in range(len(cluster_dataframe_list)-1):
        for j in range(len(cluster_dataframe_list[i])):
            x=cluster_dataframe_list[i]
            for k in range(len(cluster_dataframe_list[i+1])):
                y=cluster_dataframe_list[i+1]
                distance=math.dist(x[j],y[k])
                mini=min(mini,distance)
    return mini

def dunn_index(cluster_dataframe_list):
    numerator=inter_cluster_dist(cluster_dataframe_list)
    denominator=intra_cluster_dist(cluster_dataframe_list)
    return numerator/denominator

def cluster_wise_df(dataframe,cluster_label_list):
    cluster_df_list=[]
    for i in range(len(np.unique(cluster_label_list))):
        j=np.array(dataframe.loc[dataframe.Cluster==i].drop(["Cluster"],axis=1))
        cluster_df_list.append(j)
        j=None
    return cluster_df_list

##Silhouette score Visualization

from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans

def Silhouette_visual(data):
    
    fig, ax = plt.subplots(3, 2, figsize=(8,10))
    plt.grid(False)
    
    for i in [2, 3, 4, 5, 6, 7]:
        '''
        Create KMeans instance for different number of clusters
        '''
        km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
        q, mod = divmod(i, 2)
        '''
        Create SilhouetteVisualizer instance with KMeans instance
        Fit the visualizer
        '''
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q-1][mod])
        visualizer.fit(data)
        visualizer.ax.set_xlabel("Silhouette co-efficient values")
        visualizer.ax.set_ylabel("No of data points")
        plt.savefig('Silhouette_Plot.png', dpi=700, bbox_inches='tight')
        
        
        
 ##elbow plot
def elbow_plot(data):
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data)
        distortions.append(kmeanModel.inertia_)
        
    plt.figure(figsize=(6,3))
    plt.grid(False)
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig('Elbow_plot.png', dpi=700, bbox_inches='tight')
    plt.show()