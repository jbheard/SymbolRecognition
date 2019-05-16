import cv2
import numpy as np
from matplotlib import pyplot

# Scales a symbol within an image
def scale_symbol(img, contour, new_size):
	# Get the bounding rect of the contour
	[x, y, w, h] = cv2.boundingRect(contour)

	# Get rectangle as it's own image
	symbol = img[y:y+h, x:x+w]
	new_w = new_size[0]
	new_h = new_size[1]

	# If the image is scaled too small
	if 0 in symbol.shape:
		return None

	# Resize the symbol
	scaled = cv2.resize(symbol, (new_w, new_h), interpolation=cv2.INTER_AREA)

	# Return the scaled symbol
	return scaled

# Blit src image onto dst at position (x,y)
# Kind of like copying/pasting src onto dst
def blit_image(src, dst, x, y):
	# Get the image height, width
	h, w = src.shape[0], src.shape[1]
	# Blit src onto dst
	dst[y:y+h, x:x+w] = src
	return dst

# Get the contours from an image
def get_all_contours(img):
	# Get grayscale img
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Find threshed automatically, good for symbol detection
	ret2, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	# Morphological operation to ensure smaller portions are part of bigger character
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	thresh = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)

	# Only find external contours, characters (probably) won't be nested
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	return contours

# Main program to test
if __name__ == '__main__':
	from sys import argv

	# argv check / usage info
	if len(argv) < 2:
		print("Usage: {} image_path".format(argv[0]))
		exit(1)

	# Read the image
	img = cv2.imread(argv[1])
	if img is None:
		print("Error loading image")
		exit(1)

	# Scale all symbols down to 9x9
	new_w, new_h = 9, 9

	# Get the contours
	contours = get_all_contours(img)
	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		symbol = scale_symbol(img, contour, (new_w, new_h))
		
		# Clear the area and blit the resized symbol
		cv2.rectangle(img, (x, y), (x+w, y+h), 255, cv2.FILLED)
		blit_image(symbol, img, x, y)
	
	# Show the image
	pyplot.imshow(img)
	pyplot.show()

	print("All done")
