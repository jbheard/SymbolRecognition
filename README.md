# SymbolRecognition

A very basic python implementation of character recognition using feature vectors.

Some test cases are included with the digits 0-9, the program can be trained to work on any type of character.

It should be noted that larger feature vectors generated using higher quality images will result in more precise results. For example, if all symbols are scaled to 9x9 the results will be less accurate than if they were scaled to 16x16 pixels.

---

To run the code in this project you must have the following packages:

* [cv2](https://pypi.org/project/opencv-python/)
* [NumPy](https://pypi.org/project/numpy/)
* [matplotlib](https://pypi.org/project/matplotlib/)

---

Test cases are built into the modules, to run the tests you can do:

```
python3 symbols.py
python3 features.py
python3 identify.py
```

*symbols.py* will find all contours in the image and resize them inside a red box.

*features.py* will generate feature vectors for the 10 images corresponding to each digit 0..9 and store them in numbers.feat

*identify.py* will request an image file and identify all digits in the image.
