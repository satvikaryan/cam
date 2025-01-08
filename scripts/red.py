import cv2
import numpy as np

def detect_largest_red_contour(image):
    """
    Detect the largest red contour in the image.
    
    Parameters:
        image (numpy.ndarray): The input image in BGR format.
        
    Returns:
        tuple: (int, numpy.ndarray) where:
               - int: 1 if a significant red contour is detected, otherwise 0.
               - numpy.ndarray: Frame with the largest red contour highlighted.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])  
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    red_mask = mask1 + mask2
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        min_area_threshold = 500

        if area > min_area_threshold:
            cv2.drawContours(image, [largest_contour], -1, (0, 255, 0), 3)  
            return 1, image

    return 0, image


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("Error: Cannot open video source.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or failed to read frame.")
            break
        
        red_detected, visualized_frame = detect_largest_red_contour(frame)

        print("1" if red_detected else "0")

        cv2.imshow("Largest Red Contour Detection", visualized_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
