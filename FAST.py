import cv2
import numpy as np
import matplotlib.pyplot as plt


def is_pixel_brighter_than(pixel, threshold):
    return pixel > threshold


def is_pixel_darker_than(pixel, threshold):
    return pixel < threshold


def test_for_corner(center_pixel, pixels, threshold):
    # Convert center_pixel and threshold to float to avoid overflow
    center_pixel = float(center_pixel)
    threshold = float(threshold)

    # Use a list comprehension to get boolean arrays, then sum them
    brighter_count = np.sum([float(p) > (center_pixel + threshold) for p in pixels])
    darker_count = np.sum([float(p) < (center_pixel - threshold) for p in pixels])

    return brighter_count >= 9 or darker_count >= 9


def non_maximum_suppression(corner_candidates, distance_threshold):
    corners = []

    for corner in corner_candidates:
        x, y = corner
        is_maximum = True

        for c in corners:
            cx, cy = c
            if abs(x - cx) < distance_threshold and abs(y - cy) < distance_threshold:
                is_maximum = False
                break

        if is_maximum:
            corners.append(corner)

    return corners


def visualize_corners(image, corners):
    image_with_corners = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for corner in corners:
        cv2.circle(image_with_corners, (corner[1], corner[0]), 3, (0, 255, 0), -1)

    plt.imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))
    plt.title("Detected Corners")
    plt.show()


def find_corners(image, threshold):
    rows, cols = image.shape
    corner_candidates = []

    for x in range(3, rows - 3):
        for y in range(3, cols - 3):
            center_pixel = image[x, y]
            pixels_around = [
                image[x - 3, y], image[x - 3, y + 1], image[x - 2, y + 2],
                image[x - 1, y + 3], image[x, y + 3], image[x + 1, y + 3],
                image[x + 2, y + 2], image[x + 3, y + 1], image[x + 3, y],
                image[x + 3, y - 1], image[x + 2, y - 2], image[x + 1, y - 3],
                image[x, y - 3], image[x - 1, y - 3], image[x - 2, y - 2],
                image[x - 3, y - 1]
            ]

            if test_for_corner(center_pixel, pixels_around, threshold):
                corner_candidates.append((x, y))

    return corner_candidates


# Load the image
image = cv2.imread('/home/student/Downloads/chess.jpg', cv2.IMREAD_GRAYSCALE)

# Set threshold for corner detection
threshold = 20

# Find corners using our own FAST algorithm
corner_candidates = find_corners(image, threshold)

# Non-maximum suppression
final_corners = non_maximum_suppression(corner_candidates, distance_threshold=10)

# Visualize the results
visualize_corners(image, final_corners)
