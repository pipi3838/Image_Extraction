import os
from glob import glob
import cv2
import numpy as np
import pickle
from tqdm import tqdm
from utils import *
import Features

path = '/nfs/nas-5.1/wbcheng/cc_hw2/HW2-database-20f/'

Categories, ImagePaths = [], []
for f in glob(os.path.join(path, 'database', '*')):
    ImagePaths.append(glob(os.path.join(f, '*.jp*g')))
    Categories.append(f.split('/')[-1])

print('Total Categories: {}'.format(len(Categories)))

feature_path = os.path.join(path, 'FeatureDatabase.pkl')
if os.path.isfile(feature_path):
    print('Loading feature database ...')
    with open(feature_path, 'rb') as f:
        FeatureDatabase = pickle.load(f)
else:
    print('Initializing feature database ...')
    FeatureDatabase = {}
    for C_id, C in enumerate(Categories): 
        FeatureDatabase[C] = []
        for path in ImagePaths[C_id]:
            RawImage = LoadImage(path)
            
            CurImageDict = {}
            CurImageDict['Path'] = path
            CurImageDict['RGB'] = RawImage
            CurImageDict['Gray'] = np.expand_dims(cv2.cvtColor(RawImage, cv2.COLOR_RGB2GRAY), axis=-1)
            CurImageDict['HSV'] = cv2.cvtColor(RawImage, cv2.COLOR_RGB2HSV)
            CurImageDict['YUV'] = cv2.cvtColor(RawImage, cv2.COLOR_RGB2YUV)
            CurImageDict['Lab'] = cv2.cvtColor(RawImage, cv2.COLOR_RGB2Lab)
            
            FeatureDatabase[C].append(CurImageDict)
        print("█", end='')
    print()

#specify which feature(function), image type you want to extract
#example [['RGB', 'GlobalColorHistogram'], [], ....]
# features = [['RGB', 'GlobalColorHistogram'], ['RGB', 'LocalColorHistogram']]
# features = [['Gray', 'GaborLocalHistogram'], ['Gray', 'GetSIFT'], ['HSV', 'LocalColorHistogram'], ['HSV', 'AutoCorrelogram'], ['Gray', 'PyramidHOG'], ['RGB', 'AutoCorrelogram'], ['HSV', 'GlobalColorHistogram'], ['Gray', 'HistogramofOrientedGradients'], ['Gray', 'GetDenseSIFT']]
# features = [['HSV', 'AutoCorrelogram'], ['RGB', 'AutoCorrelogram']]
# features = [['Gray', 'PyramidHOG'], ['Gray', 'HistogramofOrientedGradients'], ['Gray', 'GaborLocalHistogram'], ['Gray', 'GetSIFT'], ['Gray', 'GetPyramidSIFT']]
# features = [['Gray', 'GaborLocalHistogram'], ['Gray', 'GaborGlobalHistogram'], ['Gray', 'HistogramofOrientedGradients'], ['Gray', 'PyramidHOG']]

features = [['HSV', 'GlobalColorHistogram'], ['HSV', 'LocalColorHistogram']]

for feature in features:
    key = feature[0] + '_' + feature[1]
    if key in FeatureDatabase[Categories[0]][0]: 
        print(key, ' feature is already extracted!')
        continue
    print('extract ', key)
    for C in tqdm(Categories):
        for ImgDict in FeatureDatabase[C]:
            extractor = getattr(Features, feature[1])
            ImgDict[key] = extractor(ImgDict[feature[0]])
    
    # with open(feature_path, 'wb') as f:
    #     pickle.dump(FeatureDatabase, f)
print('Finish Extracting')

with open(feature_path, 'wb') as f:
    pickle.dump(FeatureDatabase, f)
