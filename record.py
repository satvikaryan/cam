import cv2
import os

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))

# Define the output file path
output_file = os.path.join(script_directory, 'camera_output.avi')

# Open the default camera (camera index 0)
camera = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Get the default width and height of the frames
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))

print("Press 'q' to stop recording.")

while True:
    # Capture frames from the camera
    ret, frame = camera.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Write the frame to the output file
    out.write(frame)

    # Display the frame in a window
    cv2.imshow('Recording', frame)

    # Stop recording when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and file writer
camera.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print(f"Recording saved to {output_file}")
