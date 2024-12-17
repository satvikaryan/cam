import cv2

# Test specific camera index
camera_index = 1  
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Error: Camera at index {camera_index} could not be opened.")
    exit()

print(f"Camera at index {camera_index} is working.")
cap.release()
