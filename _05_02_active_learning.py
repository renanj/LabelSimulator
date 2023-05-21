import pandas as pd
import numpy as np

import random
import math
import warnings
warnings.filterwarnings('ignore')
# import faiss
# from faiss import StandardGpuResources
from tqdm import tqdm
import cupy as cp
import cudf

from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


from matplotlib import pyplot as plt, animation
from sklearn.datasets import make_blobs

from skactiveml.utils import MISSING_LABEL, labeled_indices, unlabeled_indices
from skactiveml.visualization import plot_utilities, plot_decision_boundary, plot_contour_for_samples

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import BaggingClassifier
from skactiveml.classifier import SklearnClassifier



from skactiveml.pool import BatchBALD





def f_batchBald(_df, _GPU_flag=True):
	
	if _GPU_flag is True:
		_temp_X_columns = [x for x, mask in zip(_df.columns.values, _df.columns.str.startswith("X")) if mask]
		X_train = _df.loc[:,_temp_X_columns].astype('float32')
		y_train = _df.loc[:,'labels'].astype('float32')		
		y_train_query = np.full(shape=y_train.shape, fill_value=MISSING_LABEL)

	else:		
		#TO-DO
		None

	 random_state = np.random.RandomState(0)

	 # Initialise the classifier.
	 clf = SklearnClassifier(BaggingClassifier(SklearnClassifier(GaussianProcessClassifier(), random_state=random_state),
	     random_state=random_state),
	     classes=np.unique(y_train),
	     random_state=random_state
	 )
	 # Initialise the query strategy.
	 qs = BatchBALD(random_state=random_state)
	 ordered_selected_samples_id = []

	
	while len(ordered_selected_samples_id) < len(X_train):

		clf.fit(X_train, y_train_query)
		# Get labeled instances.
		X_labeled = X_train[labeled_indices(y_train_query)]
		# Query the next instance/s.
		query_idx, utilities = qs.query(X=X_train, y=y_train_query, ensemble=clf, batch_size=20, return_utilities=True)
		# Label the queried instances.
		y_train_query[query_idx] = y_train[query_idx]		
		ordered_selected_samples_id.append(query_idx)


	
	return ordered_selected_samples_id	
