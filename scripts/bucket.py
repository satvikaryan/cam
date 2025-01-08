#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool
import cv2
import numpy as np

def detect_largest_red_contour(image, frame_width, frame_height, min_area_percentage=20):
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
        min_area_threshold = (min_area_percentage / 100.0) * (frame_width * frame_height)

        if area > min_area_threshold:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                center_x = int(M["m10"] / M["m00"])
                center_y = int(M["m01"] / M["m00"])

                if (frame_width * 0.4) < center_x < (frame_width * 0.6) and (frame_height * 0.4) < center_y < (frame_height * 0.6):
                    return True, red_mask, (center_x, center_y)
    return False, red_mask, None

def red_detection_node():
    rospy.init_node('red_contour_detector', anonymous=True)
    gripper_pub = rospy.Publisher('/gripper', Bool, queue_size=10)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        rospy.logerr("Cannot open video source.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rate = rospy.Rate(10)
    rospy.loginfo("Red Contour Detector Node Started.")

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logwarn("Failed to read frame.")
            break
        
        red_detected, red_mask, center = detect_largest_red_contour(frame, frame_width, frame_height)
        # Place holder for Current center

        if red_detected:
            gripper_pub.publish(Bool(data=True))
            rospy.loginfo("Red object detected in the center and covering enough area.")
            break

        gripper_pub.publish(Bool(data=False))
        # cv2.imshow("Video Feed", frame)
        # cv2.imshow("Red Mask", red_mask)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        rate.sleep()

    cap.release()
    cv2.destroyAllWindows()
    rospy.loginfo("Red Contour Detector Node Stopped.")

if __name__ == "__main__":
    try:
        red_detection_node()
    except rospy.ROSInterruptException:
        pass
