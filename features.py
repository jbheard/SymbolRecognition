import pickle, cv2
from matplotlib import pyplot
from math import exp, ceil
from symbols import scale_symbol, get_all_contours

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
def extract_feature(img, fsize):
	try: # Make sure image is in grayscale
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	except: pass
	_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	assert thresh.shape[0] == thresh.shape[1], "Image must be square"

	# Get number of features to generate, calculate block size
	fs = thresh.shape[0]
	bs = int(ceil(fs**2 / fsize)**0.5)

	feature = [0] * fsize
	k = 0
	for i in range(0, fs, bs):
		for j in range(0, fs, bs):
			if k == fsize: break
			a = thresh[i:i+bs, j:j+bs]

			# Store as ratio of black:white
			feature[k] = quantify(a.flat)
			k += 1
	return tuple(feature) # Return immutable feature

# Get the average of multiple feature vectors
def feature_avg(features, fsize):
	feat = [0]*fsize
	# For each feature vector
	for i in range(fsize):
		for f in features:
			feat[i] += f[i]
		# Get average of features
		feat[i] /= len(features)

	# Return a single feature vector
	return tuple(feat)

# Write feature array to a file
def write_features(features, fsize, fname):
	fp = open(fname, 'wb')    # open file
	pickle.dump(fsize, fp)    # write feature size
	pickle.dump(features, fp) # write feature data
	fp.close()                # close file
	return

# Read feature array from file
def read_features(fname):
	fp = open(fname, 'rb')  # open file
	fsize = pickle.load(fp) # read feature size
	data = pickle.load(fp)  # read feature data
	fp.close()              # close file
	return fsize, data


if __name__ == '__main__':
	from sys import argv

	# argv check / usage info
	if len(argv) < 2:
		print("Usage: {} feat_size".format(argv[0]))
		exit(1)

	FEAT_SIZE = int(argv[1])

	# Match each image file to the corresponding ASCII symbol
	files = [ (str(i), 'res/{}.png'.format(i)) for i in range(10) ]

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
			feat = extract_feature(symbol, FEAT_SIZE)
			features.append( feat )

		# Take the average of multiple features for each image
		# If the image only has 1 symbol, it will only use that
		all_feat.append( (val, feature_avg(features, FEAT_SIZE)) )

	# Write the features to a file
	write_features(all_feat, FEAT_SIZE, "numbers.feat")
	print("Wrote features for {} symbols".format(len(all_feat)))
	print("All done")

