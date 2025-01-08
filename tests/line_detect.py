
import cv2
import numpy as np

def detect_lines_from_video():
    # Open a video feed (0 for webcam or provide a video file path)
    cap = cv2.VideoCapture('IMG_4109.MOV')

    if not cap.isOpened():
        print("Error: Unable to access the video feed.")
        return

    # Get video properties for saving the output
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('output_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width * 2, frame_height))

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Unable to read frame.")
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Canny edge detector to find edges
        edges = cv2.Canny(blurred, 50, 150)

        # Use Hough Line Transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

        # Create a copy of the frame to draw lines
        line_frame = frame.copy()

        # Draw the lines on the line_frame
        if lines is not None:
            for rho, theta in lines[:, 0]:
                # Convert polar coordinates to Cartesian coordinates
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                # Draw the line on the line_frame
                cv2.line(line_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Combine edges and lines into a single view
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        combined = np.hstack((line_frame, edges_colored))

        # Write the combined frame to the output video
        out.write(combined)

        # Display the resulting frames
        cv2.imshow('Line Detection and Edges', combined)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer objects and close display windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_lines_from_video()