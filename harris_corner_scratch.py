import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('/home/student/Downloads/chess.jpg', cv2.IMREAD_GRAYSCALE)

# Convert the image to float and normalize
image = image.astype(np.float32)
image /= image.max()

k = 0.05
thresh = 0.5

Sx = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]])

Sy = Sx.T

# Gaussian Kernel
G = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]]) / 16

def corner_response(image, k=k):
    # compute first derivatives
    dx = cv2.filter2D(image, ddepth=-1, kernel=Sx)
    dy = cv2.filter2D(image, ddepth=-1, kernel=Sy)

    # Gaussian Filter
    A = cv2.filter2D(dx*dx, ddepth=-1, kernel=G)
    B = cv2.filter2D(dy*dy, ddepth=-1, kernel=G)
    C = cv2.filter2D(dx*dy, ddepth=-1, kernel=G)

    # compute corner response at all pixels
    return (A*B - (C*C)) - k*(A + B)*(A + B)

def get_harris_corners(image, k=k):
    # compute corner response
    R = corner_response(image, k)

    # threshold response to find corners
    corners_binary = R > thresh
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(corners_binary))

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    return cv2.cornerSubPix(image, np.float32(centroids), (9, 9), (-1, -1), criteria)

# Get Harris corners
corners = get_harris_corners(image)

# Draw corners on output image
image_out = np.dstack((image, image, image))
for (x, y) in corners:
    x = np.round(x).astype(int)
    y = np.round(y).astype(int)
    if 0 <= x < image_out.shape[1] and 0 <= y < image_out.shape[0]:  # Ensure coordinates are within image bounds
        cv2.circle(image_out, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

# Plot results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image_out)
plt.title("Harris Corners")

# Compute Harris corner response
R = corner_response(image, k)

plt.subplot(1, 2, 2)
plt.imshow(R < -0.025, cmap='gray')
plt.title("Harris Edges")

plt.show()
