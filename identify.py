import cv2
from symbols import get_all_contours, scale_symbol
from features import extract_feature, read_features
from matplotlib import pyplot

# Recognize a symbol from a list of features
# This could match multiple symbols, it will take the 
# best match out of all symbols tested
def recognize_symbol(features, img):
	# Get the feature
	feat = extract_feature(img)
	retval = None
	min = 1
	# For every feature
	for val, f in features:
		# Compare the feature to our own
		d = compare_feature(f, feat)
		if d < min:
			# We got a match!
			retval = val
			min = d
	return retval

# Compare two feature vectors using euclidean distance
def compare_feature(f1, f2):
	d = 0
	for i in range(len(f1)):
		d += (f1[i] - f2[i])**2
	return d**(0.5) # Same as sqrt(d), but no import

# Draw boxes around contour areas onto an image
def draw_boxes(img, contours, line=2, color=(0,0,255)):
	for contour in contours:
		# Get the a bounding box around the contour
		[X, Y, W, H] = cv2.boundingRect(contour)

		# Draw the bounding box around the contour
		cv2.rectangle(img, (X, Y), (X + W, Y + H), color, line)
	return

if __name__ == '__main__':
	# Read the image
	img = cv2.imread(input("Enter filename: "))
	if img is None:
		print("Error loading image")
		exit(1)
	# Get the features from the file
	features = read_features('numbers.feat')
	
	# Get all contours from the image
	contours = get_all_contours(img)

	# Loop through all contours
	for contour in contours:
		# Get the symbol and attempt to match it to a known feature
		symbol = scale_symbol(img, contour, (9,9))
		ret = recognize_symbol(features, symbol)
		
		# Output the results
		if ret is not None:
			print("identified symbol: '{}'".format(ret))
			pyplot.imshow(symbol)
			pyplot.show()
		else:
			print("Unable to identify symbol")
	print("All done")
