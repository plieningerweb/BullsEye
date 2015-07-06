import numpy as np
import matplotlib.pyplot as plt


test_image = np.random.randint(128, size=(150,2))
##fill image with random events

y_0 = 70
dy = 20
ddy = -45
for x in range(128):
	test_image[x] = [x,y_0 + x/127. * (dy + ddy * x/127.)]


threeD_hough_space = np.zeros((128,128,64), dtype=int)

for pixel in test_image:
	for dy in range(-64,64):
		for ddy in range (-63,1):
			y_0 = pixel[1] - pixel[0]/127. * (dy + ddy * pixel[0]/127.)
			if 0 <= y_0 < 128:
				threeD_hough_space[y_0,dy+64,ddy+63] += 1

[i - j for i, j in zip(
		np.unravel_index(threeD_hough_space.argmax(),threeD_hough_space.shape), (0,64,63)
	)]


twoD_hough_space = np.zeros((128,128), dtype=int)

for pixel in test_image:
	for dy in range(-64,64):
		y_0 = pixel[1] - pixel[0]/127. * (dy)
		if 0 <= y_0 < 128:
			twoD_hough_space[y_0,dy+64] += 1

[i - j for i, j in zip(
		np.unravel_index(twoD_hough_space.argmax(),twoD_hough_space.shape), (0,64)
	)]


hough_space = np.zeros((90,180), dtype=int)


#plot as image (regular gray scale)
image = np.zeros((128, 128), dtype=float)
for pixel in test_image:
	image[pixel[0],pixel[1]] += 10

(y_0,dy,ddy) = [i - j for i, j in zip(
		np.unravel_index(threeD_hough_space.argmax(),threeD_hough_space.shape), (0,64,63)
	)]
for x in range(128):
	image[x,y_0 + x/127. * (dy + ddy * x/127.)] += +5

(y_0,dy) = [i - j for i, j in zip(
		np.unravel_index(twoD_hough_space.argmax(),twoD_hough_space.shape), (0,64)
	)]
for x in range(128):
	image[x,y_0 + x/127. * (dy)] -= -5

#image[0,0] = 15

image = np.rot90(image,1)
fig = plt.figure()
vmin, vmax = -5, 15
im = plt.imshow(image, cmap=plt.get_cmap('gray'), vmin=vmin, vmax=vmax, interpolation='none')
plt.show()






for pixel in test_image:
	for thita in range(hough_space.shape[1]):
		#rho = x*cos(thita) + y*sin(thita)
		rho = (pixel[0]-64)*np.cos(thita)+(pixel[1]-64)*np.sin(thita)
		hough_space[rho,thita] += 1

np.unravel_index(hough_space.argmax(),hough_space.shape)
