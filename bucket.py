#  I will delete this shit after it starts working


#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
import cv2
import numpy as np

def detect_largest_red_contour(image, min_area_threshold=500):
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
        if cv2.contourArea(largest_contour) > min_area_threshold:
            return True
    return False

def red_detection_node():
    rospy.init_node('red_contour_detector', anonymous=True)
    gripper_pub = rospy.Publisher('/control/dropper', Bool, queue_size=10)
    rate = rospy.Rate(10)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        rospy.logerr("Cannot open video source.")
        return

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            rospy.logwarn("Failed to read frame.")
            break
        red_detected = detect_largest_red_contour(frame)
        gripper_pub.publish(Bool(data=red_detected))
        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        rate.sleep()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        red_detection_node()
    except rospy.ROSInterruptException:
        pass
