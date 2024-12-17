import cv2

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

    # Display the frame
    cv2.imshow("Camera Output", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
