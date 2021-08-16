import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal
import numpy as np

# fo1=ndi.convolve1d(x1, [1/3.0, 1/3.0, 1/3.0], output=np.float64, mode='nearest')
import scipy.ndimage as ndi


img = mpimg.imread('lena.jpg')
# imgplot = plt.imshow(img, cmap="gray")

# B = scipy.signal.convolve2d(img, [[0.2, 0.2, 0.2, 0.2, 0.2], [0,0,0,0,0]])
# plt.subplot(121)
# imgplot = plt.imshow(B, cmap="gray")

# plt.subplot(122)
# # C = scipy.signal.convolve2d(img, [[0.2],[0.2],[0.2],[0.2],[0.2]])
# C = scipy.signal.convolve2d(img, np.ones((5,1))/5)



result = np.empty((0,516), int)
result1 = np.empty((0,516), int)

KERNEL = np.array([0.2,0.2,0.2,0.2,0.2])

# Theo hàng
for x in img:
    result = np.vstack([result, scipy.signal.convolve(x, KERNEL)])
    
# Theo cột
for x in img.T:
    result1 = np.vstack([result1, scipy.signal.convolve(x, KERNEL)])

    
for x in img.T:
    result1 = np.vstack([result1, scipy.signal.convolve(x, KERNEL)])


plt.subplot(221)
imgplot = plt.imshow(result, cmap="gray")

plt.subplot(222)
imgplot = plt.imshow(result1.T, cmap="gray")

plt.subplot(223)
imgplot = plt.imshow(img, cmap="gray")

plt.show()
