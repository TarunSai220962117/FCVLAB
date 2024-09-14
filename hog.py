import cv2
import numpy as np

# Initialize the HOG descriptor with the pre-trained SVM for pedestrian detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def compute_hog_features(image):
    # Compute HoG features
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, _ = hog.compute(gray_image, winStride=(8, 8), padding=(8, 8))
    return features






def detect_humans(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform human detection
    boxes, weights = hog.detectMultiScale(gray_image, winStride=(8, 8), padding=(8, 8), scale=1.05)

    return boxes




def draw_detections(image, boxes):
    for (x, y, w, h) in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image


def main():
    # Load the image
    image = cv2.imread('/home/student/Downloads/human.jpg')

    # Detect humans in the image
    boxes = detect_humans(image)

    # Draw bounding boxes around detected humans
    result_image = draw_detections(image, boxes)

    # Show the result
    cv2.imshow('Detections', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

