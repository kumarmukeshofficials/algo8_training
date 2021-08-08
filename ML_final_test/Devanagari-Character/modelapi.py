from keras.models import load_model
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = load_model("./kerasmodels/omk1")

def predict(path):
	x1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
	(thresh, x2) = cv2.threshold(x1, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	x3 = cv2.bitwise_not(x2)
	kernel = np.ones((5,5),np.uint8)
	x4 = cv2.morphologyEx(x3, cv2.MORPH_OPEN, kernel)
	x5 = cv2.morphologyEx(x4, cv2.MORPH_CLOSE, kernel)

	x6 = Image.fromarray(x5)
	x7 = Image.new('L',(256,256), 0)
	x6w,x6h = x6.size
	x7w,x7h = x7.size
	offset = ((x7w - x6w)/2, (x7h - x6h)/2)
	x7.paste(x6,offset)
	x8 = np.asarray(x7)

	x9 = cv2.resize(x8, (64,64), interpolation = cv2.INTER_NEAREST)
	plt.imshow(x9)
	plt.show()

	z = model.predict(x9.reshape((1,64,64,1)))[0]
	#print(z, z.shape)
	z1 = np.where(z>0.5)[0]+2304
	return z1