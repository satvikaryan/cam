import cv2
import numpy as np

# Specify the camera index
camera_index = 1

# Initialize the video capture object
cap = cv2.VideoCapture(camera_index)

# Check if the camera is opened successfully
if not cap.isOpened():
    print(f"Error: Camera at index {camera_index} could not be opened.")
    exit()

print(f"Camera at index {camera_index} is now capturing...")

# Loop to continuously capture frames
while True:
    # Capture a single frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection (you can adjust thresholds here)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank image to draw contours
    mask = np.zeros_like(frame)

    if contours:
        for contour in contours:
            # Calculate the bounding rectangle of the contour
            x, y, w, h = cv2.boundingRect(contour)
            print(x,y,w,h)
            # Detect line-like contours: filter contours based on width and height
            if 50 < w < 500 and 2 < h < 50:  # Adjust thresholds as per your requirements
                # Draw the contour on the mask
                cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

                # Calculate centroid if the contour passes the filter
                moments = cv2.moments(contour)
                if moments['m00'] != 0:
                    cx = int(moments['m10'] / moments['m00'])
                    cy = int(moments['m01'] / moments['m00'])

                    # Draw the centroid on the frame
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Red dot for centroid
                    cv2.putText(frame, f"({cx}, {cy})", (cx + 10, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Combine the original frame and mask side by side for comparison
    combined_output = np.hstack((frame, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), mask))

    # Display the frames
    cv2.imshow("Original | Edge Mask | Line Mask", combined_output)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
