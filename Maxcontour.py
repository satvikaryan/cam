import cv2
import numpy as np
camera_index = 1

cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Error: Camera at index {camera_index} could not be opened.")
    exit()

print(f"Camera at index {camera_index} is now capturing...")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresholded, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_all_contours = np.zeros_like(frame)
    mask_largest_contour = np.zeros_like(frame)

    if contours:
        
        cv2.drawContours(mask_all_contours, contours, -1, (255, 255, 255), 1)

        largest_contour = max(contours, key=cv2.contourArea)

        cv2.drawContours(mask_largest_contour, [largest_contour], -1, (0, 255, 0), 2)

        moments = cv2.moments(largest_contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])

            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  
            cv2.putText(frame, f"({cx}, {cy})", (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    combined_output = np.hstack((frame, cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR), mask_all_contours, mask_largest_contour))

    cv2.imshow("Original | Thresholded | All Contours | Largest Contour", combined_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
