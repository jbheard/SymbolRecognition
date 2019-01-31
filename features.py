import pickle, cv2
from matplotlib import pyplot
from math import exp, ceil
from symbols import scale_symbol, get_all_contours

FEAT_SIZE = 12 # Number of terms in feature

# Get value in range [0,1] for feature extraction
def quantify(arr):
	b = 0
	n = len(arr)
	for i in range(n):
		if arr[i] == 0:
			b += 1
	# Use sigmoid function for ratio (gives more smooth results)
	return 1/(1 + exp(-b/n))
	#return b / n

# Extract a feature vector from an image
def extract_feature(img):
	try: # Make sure image is in grayscale
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	except: pass
	_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	assert thresh.shape[0] == thresh.shape[1], "Image must be square"
	
	# Get number of features to generate, calculate block size
	fs = thresh.shape[0]
	bs = int(ceil(fs**2 / FEAT_SIZE)**0.5)

	feature = [0] * FEAT_SIZE
	k = 0
	for i in range(0, fs, bs):
		for j in range(0, fs, bs):
			if k == FEAT_SIZE: break
			a = thresh[i:i+bs, j:j+bs]

			# Store as ratio of black:white
			feature[k] = quantify(a.flat)
			k += 1
	return tuple(feature) # Return immutable feature


# Get the average of multiple feature vectors
def feature_avg(features):
	feat = [0]*FEAT_SIZE
	# For each feature vector
	for i in range(FEAT_SIZE):
		for f in features:
			feat[i] += f[i]
		# Get average of features
		feat[i] /= len(features)
	
	# Return a single feature vector
	return tuple(feat)

# Write feature array to a file
def write_features(features, fname):
	fp = open(fname, 'wb') # open file
	pickle.dump(features, fp) # write to file
	fp.close() # close file
	return

# Read feature array from file
def read_features(fname):
	fp = open(fname, 'rb') # open file
	data = pickle.load(fp) # read from file
	fp.close() # close file
	return data


if __name__ == '__main__':
	# Match each image file to the corresponding ASCII symbol
	files = [('9', '9.png'), ('8', '8.png'), ('7', '7.png'), 
		('6', '6.png'), ('5', '5.png'), ('4', '4.png'), 
		('3', '3.png'), ('2', '2.png'), ('1', '1.png'), 
		('0', '0.png')]

	all_feat = []
	for val, fname in files:
		# Read the image
		img = cv2.imread(fname)
		if img is None:
			print("Error loading image {}".format(fname))
			continue
		
		# Get all contours
		contours = get_all_contours(img)
		features = []
		for contour in contours:
			# Get the symbol from the image
			symbol = scale_symbol(img, contour, (FEAT_SIZE, FEAT_SIZE))
			# Get the feature vector and add it to the list
			feat = extract_feature(symbol)
			features.append( feat )
		
		# Take the average of multiple features for each image
		# If the image only has 1 symbol, it will only use that
		all_feat.append( (val, feature_avg(features)) )

	# Write the features to a file
	write_features(all_feat, "numbers.feat")
	print("Wrote features for {} symbols".format(len(all_feat)))
	print("All done")

