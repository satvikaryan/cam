import cv2
import numpy as np

camera_index = 0
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

    height, width = frame.shape[:2]
    border_margin = 10

    if contours:
        cv2.drawContours(mask_all_contours, contours, -1, (255, 255, 255), 1)
        valid_contours = [
            contour for contour in contours
            if not np.any(
                (contour[:, :, 0] < border_margin) | 
                (contour[:, :, 0] > width - border_margin) |
                (contour[:, :, 1] < border_margin) | 
                (contour[:, :, 1] > height - border_margin)
            )
        ]

        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            cv2.drawContours(mask_largest_contour, [largest_contour], -1, (0, 255, 0), 2)

            mask_gray = cv2.cvtColor(mask_largest_contour, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(mask_gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

            if lines is not None:
                lines = [line[0] for line in lines]
                parallel_lines = []
                l_shape_found = False
                parallel_centers = []
                l_shape_center = None

                parallel_tolerance = np.pi / 36  # ~5 degrees
                l_shape_tolerance = np.pi / 18  # ~10 degrees

                for i in range(len(lines)):
                    rho1, theta1 = lines[i]
                    for j in range(i + 1, len(lines)):
                        rho2, theta2 = lines[j]
                        angle_diff = abs(theta1 - theta2)

                        if angle_diff < parallel_tolerance:
                            parallel_lines.append((lines[i], lines[j]))
                            x1, y1 = rho1 * np.cos(theta1), rho1 * np.sin(theta1)
                            x2, y2 = rho2 * np.cos(theta2), rho2 * np.sin(theta2)
                            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            parallel_centers.append((center_x, center_y))

                        # Check for L-shape
                        elif abs(angle_diff - np.pi / 2) < l_shape_tolerance:
                            l_shape_found = True
                            x1, y1 = rho1 * np.cos(theta1), rho1 * np.sin(theta1)
                            x2, y2 = rho2 * np.cos(theta2), rho2 * np.sin(theta2)
                            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                            l_shape_center = (center_x, center_y)

                for rho, theta in lines:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                if parallel_lines:
                    print("Detected parallel lines.")
                    for center in parallel_centers:
                        cv2.circle(frame, center, 5, (0, 255, 255), -1)
                    cv2.putText(frame, "Parallel Lines", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if l_shape_found:
                    print("Detected L-shape.")
                    if l_shape_center:
                        cv2.circle(frame, l_shape_center, 5, (0, 0, 255), -1)
                    cv2.putText(frame, "L-Shape Detected", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    combined_output = np.hstack((
        frame,
        cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR),
        mask_all_contours,
        mask_largest_contour
    ))

    cv2.imshow("Original | Thresholded | All Contours | Largest Contour", combined_output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
