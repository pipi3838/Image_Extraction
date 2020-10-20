import cv2
import numpy as np
from scipy.stats import skew
from scipy.spatial import distance as dist
import skimage.feature

def GlobalColorHistogram(Image, channels=[0, 1, 2], n_bins=[36, 2, 2], ranges=[0, 256, 0, 256, 0, 256]):
	'''
	Calculate the global color histogram of an image
	Inputs:
		Image: Image with shape (width, height, n_channels)
		channels: Channels to compute color histogram
		n_bins: Number to quantify the channels
		ranges: Ranges for each channel
	Outputs:
		Hist: Global color histogram of the image
	'''
	Hist = cv2.calcHist([Image], channels, None, n_bins, ranges).flatten()
	Hist = Hist / np.sum(Hist)
	return Hist

def LocalColorHistogram(Image, n_rows=8, n_cols=8, channels=[0, 1, 2], n_bins=[36, 2, 2], ranges=[0, 256, 0, 256, 0, 256]):
	'''
	Calculate the local (regional) color histogram of an image
	Inputs:
		Image: Image with shape (width, height, n_channels)
		n_rows: Number of rows to cut the image
		n_cols: Number of columns to cut the image
		channels: Channels to compute color histogram
		n_bins: Number to quantify the channels
		ranges: Ranges for each channel
	Outputs:
		Hist: Local color histogram of the image
	'''
	Width, Height, Channel = Image.shape
	dW, dH = Width // n_rows, Height // n_cols
	Hist = []
	for r in range(n_rows):
		for c in range(n_cols):
			Block = Image[dW*r:dW*(r+1), dH*c:dH*(c+1)]
			Hist.append(GlobalColorHistogram(Block, channels, n_bins, ranges))
	Hist = np.array(Hist)
	return Hist

def GlobalColorMoment(Image, channels=[0, 1, 2], Weights=[1, 2, 1]):
	'''
	Calculate the global color moment of an image
	Inputs:
		Image: Image with shape (width, height, n_channels)
		channels: Channels to compute color moment
		Weights: Weight for mean, variance, skew
	Outputs:
		Moment: Global color moment of the image
	'''
	Image = Image / 255.
	Moment = []
	for i in channels:
		Channel = Image[:,:,i].flatten()
		Mean = np.mean(Channel) * Weights[0]
		Var = np.var(Channel) * Weights[1]
		Skew = skew(Channel) * Weights[2]
		Moment.append([Mean, Var, Skew])
	Moment = np.array(Moment).flatten()
	return Moment

def LocalColorMoment(Image, n_rows=8, n_cols=8, channels=[0, 1, 2], Weights=[0.01, 0.001, 3.0]):
	'''
	Calculate the local (regional) color moment of an image
	Inputs:
		Image: Image with shape (width, height, n_channels)
		n_rows: Number of rows to cut the image
		n_cols: Number of columns to cut the image
		channels: Channels to compute color moment
		Weights: Weight for mean, variance, skew
	Outputs:
		Moment: Local color moment of the image
	'''
	Width, Height, Channel = Image.shape
	dW, dH = Width // n_rows, Height // n_cols
	Moment = []
	for r in range(n_rows):
		for c in range(n_cols):
			Block = Image[dW*r:dW*(r+1), dH*c:dH*(c+1)]
			Moment.append(GlobalColorMoment(Block, channels, Weights))
	Moment = np.array(Moment)
	return Moment

def InBound(Shape, Index):
	'''
	Check if the Index is in bound of image
	Inputs:
		Shape: Shape of the image
		Index: The index to check
	Outputs:
		Valid: True id index is in bound, otherwize False
	'''
	if (Index[0] >= 0) and (Index[1] >= 0) and (Index[0] < Shape[0]) and (Index[1] < Shape[1]):
		Valid = True
	else:
		Valid = False
	return Valid

def GetNeighborColors(Image, Center, Distance):
	'''
	Get the color of neighbors that are certain distance away from center
	Inputs:
		Image: Image after quantization
		Center: The query center index
		Distance: The desired distance for neighbors
	Outputs:
		Colors: Colors of valid neighbors
	'''
	Shape = Image.shape
	X, Y = Center
	Colors = []
	P1 = [X-Distance, Y]
	if InBound(Shape, P1):
		Colors.append(Image[P1[0]][P1[1]])
	P2 = [X+Distance, Y]
	if InBound(Shape, P2):
		Colors.append(Image[P2[0]][P2[1]])
	P3 = [X, Y-Distance]
	if InBound(Shape, P3):
		Colors.append(Image[P3[0]][P3[1]])
	P4 = [X, Y+Distance]
	if InBound(Shape, P4):
		Colors.append(Image[P4[0]][P4[1]])
	P5 = [X-Distance, Y-Distance]
	if InBound(Shape, P5):
		Colors.append(Image[P5[0]][P5[1]])
	P6 = [X-Distance, Y+Distance]
	if InBound(Shape, P6):
		Colors.append(Image[P6[0]][P6[1]])
	P7 = [X+Distance, Y-Distance]
	if InBound(Shape, P7):
		Colors.append(Image[P7[0]][P7[1]])
	P8 = [X+Distance, Y+Distance]
	if InBound(Shape, P8):
		Colors.append(Image[P8[0]][P8[1]])
	return Colors

def GetProb(Image, Center, Distance):
	'''
	Get the probability of pixels that are in certain distance away has the same color as the center
	Inputs:
		Image: Image after quantization
		Center: The query center index
		Distance: The desired distance for neighbors
	Outputs:
		Prob: Probability of neighbor color equal to center color
	'''
	Shape = Image.shape
	X, Y = Center
	TargetColor = Image[X][Y]
	NeighborColors = np.array(GetNeighborColors(Image, Center, Distance))
	Prob = np.mean(NeighborColors == TargetColor)
	return Prob
				
def AutoCorrelogram(Image, channels=[0, 1, 2], n_bins=[16, 2, 2], ranges=[0, 180, 0, 256, 0, 256], Distances=[1, 2, 4, 8, 16, 32, 64]):
	'''
	Calculate the auto-correlogram of an image
	Inputs:
		Image: Image with shape (width, height, n_channels)
		channels: Channels to compute auto-correlogram
		n_bins: Number of bins for each channel
		ranges: Range of value of each channel
		Distances: Distances to compute auto-correlogram
	Outputs:
		Correlogram: The auto-correlogram
	'''
	W, H, C = Image.shape
	temp = []
	for i in channels:
		Channel = np.array(Image[:, :, i])
		Channel -= ranges[2*i]
		Channel = Channel // ((ranges[2*i+1] - ranges[2*i]) / n_bins[i])
		temp.append(Channel)
	temp = np.array(temp).astype(int)
	Image = np.zeros((W, H), int)
	Mult = 1
	for i in range(temp.shape[0]):
		Image += temp[i, :, :] * Mult
		Mult *= n_bins[i]
	Correlogram = []
	for d in Distances:
		ColorProbs = [[] for _ in range(np.prod(n_bins))]
		for w in range(W):
			for h in range(H):
				ColorProbs[Image[w, h]].append(GetProb(Image, [w, h], d))
		ColorProbs = [(sum(x) / len(x)) if len(x) > 0 else 1e-3 for x in ColorProbs]
		Correlogram.append(ColorProbs)
	Correlogram = np.array(Correlogram).flatten()
	return Correlogram

def GaborFilteredImages(Image, Kernels=[16], Sigmas=[1.0, 2.0, 4.0, 8.0], Thetas=[0, np.pi/8, np.pi/4, np.pi*3/8, np.pi/2, np.pi*5/8, np.pi*3/4, np.pi*7/8]):
	'''
	Generate images after gabor filters
	Inputs:
		Image: Gray scale image
		Kernels: Kernel sizes for gabor filters
		Sigmas: Sigmas for gabor filters
		Thetas: Orientation of the filters
	Outputs:
		FilteredImages: Images after gabor filters
	'''
	Image = Image.squeeze()
	GaborFilters = []
	for K in Kernels:
		for t in Thetas:
			for s in Sigmas:
				Filter = cv2.getGaborKernel((K, K), s, t, np.pi/2, 0.5, 0, ktype=cv2.CV_32F)
				GaborFilters.append(Filter / np.sum(Filter))
	FilteredImages = []      
	for F in GaborFilters:
		FilteredImage = cv2.filter2D(Image, cv2.CV_8UC3, F)
		FilteredImages.append(np.expand_dims(FilteredImage, axis=-1))
	FilteredImages = np.array(FilteredImages)
	return FilteredImages

def ExtractGaborFeature(GaborImages):
	'''
	Extract features from gabor filter filtered images
	Imputs:
		GaborImages: List of images after gabor filters
	Outputs:
		Features: Features extracted from images
	'''
	Features = []
	for i in GaborImages:
		i = i.squeeze()
		i = i / 255.
		LocalEnergy = np.mean(np.square(i))
		MeanAmplitude = np.mean(i)
		VarAmplitude = np.var(i)
		Features.append(np.array([LocalEnergy, MeanAmplitude, VarAmplitude]))
	Features = np.array(Features).T
	return Features

def GaborGlobalHistogram(GaborImages, channels=[0], n_bins=[32], ranges=[0, 256]):
	'''
	Calculate the global histogram of gabor images
	Inputs:
		GaborImages: List of images after gabor filters
		channels: Channels to compute color histogram
		n_bins: Number to quantify the channels
		ranges: Ranges for each channel
	Outputs:
		GaborHist: List of global histogram of the gabor images
	'''
	GaborHist = []
	for Image in GaborImages:
		Hist = GlobalColorHistogram(Image, channels=channels, n_bins=n_bins, ranges=ranges)
		GaborHist.append(Hist)
	GaborHist = np.array(GaborHist)
	return GaborHist

def GaborLocalHistogram(Images, n_rows=8, n_cols=8, channels=[0], n_bins=[32], ranges=[0, 256]):
	'''
	Calculate the local (regional) histogram of gabor images
	Inputs:
		GaborImages: List of images after gabor filters
		n_rows: Number of rows to cut the image
		n_cols: Number of columns to cut the image
		channels: Channels to compute color histogram
		n_bins: Number to quantify the channels
		ranges: Ranges for each channel
	Outputs:
		GaborHist: List of local histogram of the gabor images
	'''
	GaborImages = GaborFilteredImages(Images)
	GaborHist = []
	for Image in GaborImages:
		Hist = LocalColorHistogram(Image, n_rows, n_cols, channels, n_bins, ranges)
		GaborHist.append(Hist)
	GaborHist = np.array(GaborHist).reshape(len(GaborImages) * n_rows * n_cols, -1)
	return GaborHist

def LocalBinaryPattern(Image, radius=3, points=64):
	'''
	Calculate local binary pattern histogram
	Inputs:
		Image: Gray scale image
		radius: RRadius of circle (spatial resolution of the operator)
		points: Number of circularly symmetric neighbour set points (quantization of the angular space)
	Output:
		Hist: The local binary pattern histogram
	'''
	Image = Image.squeeze()
	LBP = np.expand_dims(skimage.feature.local_binary_pattern(Image, points, radius, method='uniform'), axis=-1).astype(np.float32)
	Hist = cv2.calcHist([LBP], [0], None, [points+2], [0, points+2]).flatten()
	Hist = Hist / np.sum(Hist)
	return Hist

def GridLocalBinaryPattern(Image, n_rows=8, n_cols=8, radius=3, points=64):
	'''
	Calculate the local (regional) color histogram of an image
	Inputs:
		Image: Gray scale image
		n_rows: Number of rows to cut the image
		n_cols: Number of columns to cut the image
		radius: Radius of circle (spatial resolution of the operator)
		points: Number of circularly symmetric neighbour set points (quantization of the angular space)
	Outputs:
		Hist: Grid local color histogram of the image
	'''
	Width, Height, Channel = Image.shape
	dW, dH = Width // n_rows, Height // n_cols
	Hist = []
	for r in range(n_rows):
		for c in range(n_cols):
			Block = Image[dW*r:dW*(r+1), dH*c:dH*(c+1)]
			Hist.append(LocalBinaryPattern(Block, radius, points))
	Hist = np.array(Hist)
	return Hist

def HistogramofOrientedGradients(Image, n_rows=8, n_cols=8, orientations=64):
	'''
	Calculate the histogram of oriented gradients (HOG) an image
	Inputs:
		Image: Gray scale image
		n_rows: Number of rows to cut the image
		n_cols: Number of columns to cut the image
		orientaions: Number of orientations
	Outputs:
		HOG: Histogram of oriented gradients of the image
	'''
	Image = Image.squeeze()
	Width, Height = Image.shape
	dW, dH = Width // n_rows, Height // n_cols
	HOG = skimage.feature.hog(Image, orientations=orientations, pixels_per_cell=(dW, dH), cells_per_block=(1, 1), feature_vector=False)
	HOG = np.reshape(HOG, (-1, orientations))
	GlobalGradient = np.sum(HOG, axis=0).reshape(1, -1)
	return HOG

def PyramidHOG(Image, n_rows=[1, 2, 4, 8, 16, 32], n_cols=[1, 2, 4, 8, 16, 32], orientations=64):
	'''
	Calculate the HOG of an image in different resolutions
	Inputs:
		Image: Gray scale image
		n_rows: Number of rows to cut the image
		n_cols: Number of columns to cut the image
		orientaions: Number of orientations
	Outputs:
		PyramidHOG: Pyramid histogram of oriented gradients of the image
	'''
	Image = Image.squeeze()
	PyramidHOG_ = []
	for i in range(len(n_rows)):
		HOG = HistogramofOrientedGradients(Image, n_rows[i], n_cols[i], orientations) * (1 / n_rows[i]**2)
		PyramidHOG_.append(HOG)
	PyramidHOG = PyramidHOG_[0]
	for PHOG in PyramidHOG_[1:]:
		PyramidHOG = np.concatenate((PyramidHOG, PHOG), axis=0)
	return PyramidHOG

def ShapeIndex(Image, Sigmas=[0.5, 1, 2, 4, 8, 16]):
	'''
	Calculate the image index of an image with different sigmas, then calculate histogram
	Inputs:
		Image: Gray scale image
		Sigmas: List of sigmas
	Outputs:
		ShapeIndexHist: The shape index histogram
	'''
	Image = Image.squeeze()
	ShapeIndexHist = []
	for s in Sigmas:
		ShapeIndex = np.expand_dims(skimage.feature.shape_index(Image, sigma=s), axis=-1).astype(np.float32)
		# Flat regions are difined as nan
		ShapeIndex[np.isnan(ShapeIndex)] = 1.3
		Hist = GlobalColorHistogram(ShapeIndex, channels=[0], n_bins=[10], ranges=[-1.125, 1.375])
		# Since falt regions are mostly background, we give a small weight to it
		Hist[-1] *= 0.1
		Hist = Hist / np.sum(Hist)
		ShapeIndexHist.append(Hist)
	ShapeIndexHist = np.array(ShapeIndexHist)
	return ShapeIndexHist

def GetSIFT(Image, limit=None):
	'''
	Detect and compute SIFT descriptors of an image
	Inputs:
		Image: Gray scale image
		limit: Maximun number of keypoints to detect
	Outputs:
		des: The keypoints SIFT descriptors
	'''
	# Use default sigma(1.6) would find 0 keypoints in a tennis image QAQ
	if limit is None:
		sift = cv2.xfeatures2d.SIFT_create()
	else:
		sift = cv2.xfeatures2d.SIFT_create(limit)
	kp, des = sift.detectAndCompute(Image, None)
	return des

def GetDenseSIFT(Image, n_rows=32, n_cols=32):
	'''
	Compute SIFT discriptors on grids
	Inputs:
		Image: Gray scale image
		n_rows: Number of rows to cut the image
		n_cols: Number of columns to cut the image
	Outputs:
		des: The grid SIFT descriptors
	'''
	sift = cv2.xfeatures2d.SIFT_create()
	Width, Height, Channel = Image.shape
	dW, dH = Width // n_rows, Height // n_cols
	KPs = []
	for x in range(0, Width, dW):
		for y in range(0, Height, dH):
			kp = cv2.KeyPoint(x, y, (dW+dH)/2)
			KPs.append(kp)
	kp, des = sift.compute(Image, KPs)
	return des

def GetPyramidSIFT(Image, n_rows=[1, 2, 4, 8, 16, 32, 64], n_cols=[1, 2, 4, 8, 16, 32, 64]):
	'''
	Compute SIFT discriptors on different sizes of grids
	Inputs:
		Image: Gray scale image
		n_rows: List of number of rows to cut the image
		n_cols: List of number of columns to cut the image
	Outputs:
		Pyramiddes: The pyramid grid SIFT descriptors
	'''
	Pyramiddes_ = []
	for i in range(len(n_rows)):
		des = GetDenseSIFT(Image, n_rows=n_rows[i], n_cols=n_cols[i])
		Pyramiddes_.append(des)
	Pyramiddes = Pyramiddes_[0]
	for des in Pyramiddes_[1:]:
		Pyramiddes = np.concatenate((Pyramiddes, des), axis=0)
	return Pyramiddes