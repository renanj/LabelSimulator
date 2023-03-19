import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#K-means
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs
from sklearn import datasets

from sklearn import neighbors, datasets
from sklearn.manifold import TSNE

import random
# from celluloid import Camera

import warnings
warnings.filterwarnings('ignore')

import faiss




def func_NSN(_df, _columns=None, _neighbors=5):

  if _columns == None:
      _columns = list(_df.loc[:,_df.columns.str.startswith("X")].columns)


  X = np.ascontiguousarray(_df[_columns].values.astype('float32'))

  d = X.shape[1]
  index = faiss.IndexFlatL2(d)
  index.add(X)
  
  
  distances, indices = index.search(X, _neighbors)
  return distances, indices    

def func_B(x, Vt):
    # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
    # Vt = labeled/trained feature vector 
    
    _min_dist = None
    
    for i in range(len(Vt)):         
        _d_temp = np.linalg.norm(x-Vt[i])
        
        #print(i, "   =  ", Vt[i])
        #print(_d_temp)
        #print("---------\n")
        
        if _min_dist == None:
            _min_dist = _d_temp 
        else:
            if _d_temp < _min_dist:
                _min_dist = _d_temp
            else:
                None
    
    return _min_dist
    
    
    
def func_SPB(Vc, Vt, sample_id_list, print_count = 25):
    
    # Vc = feature vector candidates
    # Vt = training Vector samples (labeled) 
    
    print_count_acc = 0
    _max_distance = None
    for i in range(len(Vc)):  
        if i % print_count == 0:
            print_count_acc = print_count_acc + print_count            
            # print('{:,.2%}'.format((print_count_acc / len(Vc))))
            print(str(print_count_acc) + '/' + str(len(Vc)))

        _dist = func_B(x=Vc[i], Vt=Vt)
        
        if _max_distance is None:
            _max_distance = _dist
            candidate = i
        else:
            if _max_distance < _dist:
                _max_distance = _dist
                candidate = i
            else:
                None
                    

    candidate = sample_id_list[candidate]
    #print(candidate)    
    return _max_distance, candidate





def func_CLU(Vc, columns):
    
    
    kmeans = KMeans(n_clusters=3, random_state=0).fit(Vc[columns].values)      
    
    Vc['kmeans_labels'] = None
    Vc['kmeans_labels'] = kmeans.labels_
    
    return Vc, kmeans.cluster_centers_
   
    
    
def func_DEN(df, columns, sample_id, k=5):
    
    #Vc = dataframe
    
    
    NSN = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    NSN.fit(df[columns])
    distances, indices = NSN.kneighbors()
    
    _index = df[df['sample_id'] == sample_id].index[0]
    DEN = (distances[_index].sum() * distances[_index].sum() / len(distances[_index])) * - 1
    return DEN


def func_OUT(df, columns, sample_id, threshold, k=5):

        return func_DEN(df, columns, sample_id, k) * -1
    
    
# def func_CE(_df, _feature_columns, _predicated_class=True, k=None):
    
  
