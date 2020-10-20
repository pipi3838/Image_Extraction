import os
import glob
import cv2
import pickle
import numpy as np
from utils import *

path = '/nfs/nas-5.1/wbcheng/cc_hw2/HW2-database-20f/'
feature_path = os.path.join(path, 'FeatureDatabase.pkl')

with open(feature_path, 'rb') as f:
    FeatureDatabase = pickle.load(f)

# features = [['HSV', 'AutoCorrelogram'], ['RGB', 'AutoCorrelogram'], ['Gray', 'PyramidHOG'], ['HSV', 'LocalColorHistogram'], ['Gray', 'GaborLocalHistogram'], ['Gray', 'GetSIFT']]
# features_list = [f[0] + '_' + f[1] for f in features]
# RunExperiment(FeatureDatabase, 
#               FeatureList=features_list, 
#               MetricList=['cityblock', 'cityblock', 'cityblock', 'cityblock', 'cityblock', 'match'],
#               WeightList=[1.4, 1.2, 0.6, 0.4, 0.06, 1.0])

features = [['HSV', 'AutoCorrelogram'], ['RGB', 'AutoCorrelogram'], ['Gray', 'PyramidHOG'], ['HSV', 'LocalColorHistogram'], ['Gray', 'GaborLocalHistogram']]
features_list = [f[0] + '_' + f[1] for f in features]
RunExperiment(FeatureDatabase, 
              FeatureList=features_list, 
              MetricList=['cityblock', 'cityblock', 'cityblock', 'cityblock', 'cityblock'],
              WeightList=[1.0, 1.0, 0.4, 0.4, 0.06])

# features = [['Gray', 'GetSIFT']]
# features_list = [f[0] + '_' + f[1] for f in features]
# RunExperiment(FeatureDatabase, 
#               FeatureList=features_list, 
#               MetricList=['match'],
#               WeightList=[1.0])
