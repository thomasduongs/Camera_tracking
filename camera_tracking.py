import rospy
import sys, time, os

sys.path.insert(1, os.path.abspath("."))
from lib.params import VISION_IMAGE_TOPIC
from lib.board_tracker import BoardTracker
from src.fetch_controller_python.fetch_robot import FetchRobot
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class PoseTracker:
    def __init__(self):
        self.cameraPoseTilt = 0
        self.cameraPosePan = 0

bridge = CvBridge()

def execute(T): 
    dx, dy, dz = T
    x_ratio = dx/dz
    y_ratio = dy/dz

    xThreshold = 0.364
    yThreshold = 0.268

    alphaHorizontal = 0
    alphaVertical = 0

    if x_ratio > xThreshold:
        alphaHorizontal = .005
    elif x_ratio < -xThreshold:
        alphaHorizontal = -.005
    if y_ratio > yThreshold:
        alphaVertical = .005
    elif y_ratio < -yThreshold:
        alphaVertical = -.005

    newHorizontal = ptk.cameraPosePan + alphaHorizontal
    newVertical = ptk.cameraPoseTilt + alphaVertical

    if np.abs(newHorizontal) > 1.5708:
        newHorizontal = ptk.cameraPosePan

    if newVertical > 1.5708 or newVertical < -0.785:
        newVertical = ptk.cameraPoseTilt

    inputMatrix = [[newHorizontal, newVertical], [0.0 for _ in range(2)], [0.0 for _ in range(2)]]
    robot.lookAt(inputMatrix)

    ptk.cameraPosePan = newHorizontal
    ptk.cameraPoseTilt = newVertical

def image_callback(img_msg): 
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))

    try:
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        results = tracker.detect(gray)

        M, _, T = tracker.getTransformation(results[0])
        execute(T)
        publisher.publish(my_msg.reshape(-1))
    except:
        print("NO APRILTAG DETECTED")
        my_msg = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        publisher.publish(my_msg)
    
if __name__ == '__main__':
    rospy.init_node('head_movement_tracking')

    robot = FetchRobot()
    tracker = BoardTracker()
    ptk = PoseTracker()

    sub_image = rospy.Subscriber(VISION_IMAGE_TOPIC, Image, image_callback)
    publisher = rospy.Publisher('board2cam', numpy_msg(Floats), queue_size=10)

    while not rospy.is_shutdown():
        rospy.spin()


 