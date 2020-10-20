import cv2
import numpy as np
import time
from scipy.stats import skew
from scipy.spatial import distance as dist


def LoadImage(path, size=(320, 320)):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img

def GetMatchDistance(Features):
	'''
	Get Matches of features using flann based matcher
	Inputs:
		Features: List of features of images
	Outputs:
		Matches: Number of matches that passed Lowe's ratio test, multiplied by -1 
	'''
	Matches = np.zeros((len(Features), len(Features)), np.float32)
	for i in range(len(Features)):
		for j in range(i, len(Features)):
			Feature_i, Feature_j = Features[i], Features[j]
			if (Feature_i is None) or (Feature_j is None):
				Matches[i, j] = 0
				continue
			if min(len(Feature_i), len(Feature_j)) < 2:
				Matches[i, j] = 0
				continue
			FLANN_INDEX_KDTREE = 1
			index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
			search_params = dict(checks=50)
			flann = cv2.FlannBasedMatcher(index_params, search_params)
			matches = flann.knnMatch(Feature_i, Feature_j, k=2)
			# Apply ratio test
			ratio = 0.75
			good_matches = [m1 for m1, m2 in matches if m1.distance < ratio * m2.distance]
			if len(good_matches) == 0:
				Matches[i, j] = 0
			else:
				Matches[i, j] = len(good_matches)
	Matches = -(Matches + Matches.T)
	return Matches

def GetMAP(Features, Id2Label, Metrics, Weights=None):
	'''
	Calculate overall MAP with leave 1 out fashion
	Inputs:
		Features: List of features of images
		Id2Label: A dictionary that maps id to label
		Metrics: List of metrics for calculating distance
		Weights: Weight for each feature
	Outputs:
		MAP: Overall MAP
		CMAP: Category MAP
	'''
	SumAP = 0.0
	CategoryAP = {}
	NumofFeatures = len(Features)
	NumofImages = Features[0].shape[0]

	# Get Distance
	Distances = np.zeros((NumofFeatures, NumofImages, NumofImages), np.float32)
	for n in range(NumofFeatures):
		CurFeature = Features[n]
		if Metrics[n] == 'match':
			Distance_ = GetMatchDistance(CurFeature)
			Distances[n] = Distance_
		elif len(CurFeature.shape) == 2:
			Distance_ = dist.squareform(dist.pdist(CurFeature, metric=Metrics[n]))
			Distances[n] += Distance_
		elif len(CurFeature.shape) == 3:
			for i in range(CurFeature.shape[1]):
				Distance_ = dist.squareform(dist.pdist(CurFeature[:,i,:], metric=Metrics[n]))
				Distances[n] += Distance_

	# Normalize each feature distance
	for i in range(NumofFeatures):
		D = Distances[i]
		Mean = np.mean(D, axis=1)
		Var = np.var(D, axis=1)
		Distances[i] = ((D.T - Mean) / Var).T

	# Apply weight to each distance
	if Weights is None:
		Distance = np.sum(Distances, axis=0)
	else:
		Distance = np.zeros((NumofImages, NumofImages), np.float32)
		for i in range(NumofFeatures):
			Distance += Distances[i] * Weights[i]
	
	# Calculate MAP
	for i in range(NumofImages):
		TargetLabel = Id2Label[i]
		Rank = np.argsort(Distance[i])
		TP = 0.0
		SumPrecision = 0.0
		for j in range(1, NumofImages):
			Id = Rank[j]
			if Id2Label[Id] == TargetLabel:
				TP += 1.0
				SumPrecision += TP / j
		AP = SumPrecision / TP
		SumAP += AP
		if TargetLabel in CategoryAP:
			CategoryAP[TargetLabel].append(AP)
		else:
			CategoryAP[TargetLabel] = [AP]
	MAP = SumAP / NumofImages
	CMAP = {}
	for Label in CategoryAP:
		CMAP[Label] = sum(CategoryAP[Label]) / len(CategoryAP[Label])
	return MAP, CMAP

def GetFeatures(FeatureDatabase, FeatureNames):
	'''
	Get features from FeatureDatabase
	Inputs:
		FeatureDatabase: The feature database
		FeatureNames: List of feature names
	Outputs:
		Features: The desired features
		Id2Label: A dictionary that maps id to label
		Id2Image: A dictionary that maps id to image
	'''
	Id2Label = {}
	Id2Image = {}
	Features = [[] for _ in range(len(FeatureNames))]
	Id = 0
	for Label in FeatureDatabase:
		for Image in FeatureDatabase[Label]:
			for i in range(len(FeatureNames)):
				Features[i].append(Image[FeatureNames[i]])
			Id2Label[Id] = Label
			Id2Image[Id] = Image['RGB']
			Id += 1
	Features = [np.array(x) for x in Features]
	return Features, Id2Label, Id2Image

def RunExperiment(FeatureDatabase, FeatureList, MetricList, WeightList=None):
	'''
	Run Experiment and print MAP, best 2 category & worst 2 category
	Inputs:
		FeatureDatabase: The feature database
		FeatureList: List of feature names (key for FeatureDatabase)
		MetricList: List of distance metrics for each feature
		WeightList: Weight for each feature, if None, simply sum up all metrics
	Outputs:
		None
	'''
	# Check if the arguments are valid
	if len(FeatureList) != len(MetricList):
		print("ERROR: FeatureList is not the same length of MetricList")
		return
	if WeightList is not None:
		if len(FeatureList) != len(WeightList):
			print("ERROR: FeatureList is not the same length of WeightList")
			return

	StartTime = time.time()
	Features, Id2Label, Id2Image = GetFeatures(FeatureDatabase, FeatureList)
	MAP, CMAP = GetMAP(Features, Id2Label, Metrics=MetricList, Weights=WeightList)
	Time = time.time() - StartTime
	print("MAP: %8f - Time: %8fs" % (MAP, Time))
	Labels = []
	CMAPs = []
	for Label in CMAP:
	    Labels.append(Label)
	    CMAPs.append(CMAP[Label])
	Labels = np.array(Labels)
	CMAPs = np.array(CMAPs)
	Rank = np.argsort(CMAPs)
	print("Best: %s(%8f)" % (Labels[Rank[-1]], CMAPs[Rank[-1]]), end="")
	print(", %s(%8f)" % (Labels[Rank[-2]], CMAPs[Rank[-2]]))
	print("Worst: %s(%8f)" % (Labels[Rank[0]], CMAPs[Rank[0]]), end="")
	print(", %s(%8f)" % (Labels[Rank[1]], CMAPs[Rank[1]]))
	return 