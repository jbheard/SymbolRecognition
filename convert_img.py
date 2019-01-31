import cv2
from matplotlib import pyplot

if __name__ == '__main__':
	dim = (128, 64) # Screen dimensions are 64x128
	
	# Read the image
	img = cv2.imread(input("Enter filename: "))
	if img is None:
		print("Error loading image")
		exit(1)
	
	# Get grayscale image
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Find threshed automatically
	_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	
	thresh = cv2.resize(thresh, dim, interpolation=cv2.INTER_AREA)

	pyplot.imshow(thresh)
	pyplot.show()
	
	save = input("Keep image(Y/n)? ")
	if save.lower() == 'y':
		fname = input("Enter file name: ")
		cv2.imwrite(thresh, fname)
