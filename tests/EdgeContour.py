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
    
    edges = cv2.Canny(gray, 100, 200)  

    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    contour_output = frame.copy()
    cv2.drawContours(contour_output, contours, -1, (0, 255, 0), 2)  

    
    edge_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  
    combined = np.hstack((frame, edge_colored, contour_output))

    
    cv2.imshow("Original | Edge Mask | Contours", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
