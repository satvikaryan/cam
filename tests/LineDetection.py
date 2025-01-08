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

    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(frame)

    if contours:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            print(x,y,w,h)
            if 50 < w < 500 and 2 < h < 50:  
                cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

                # Calculate centroid if the contour passes the filter
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])

                    # Draw the centroid on the frame
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1) 
                    cv2.putText(frame, f"({cx}, {cy})", (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    combined_output = np.hstack((frame, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), mask))

    cv2.imshow("Original | Edge Mask | Line Mask", combined_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
