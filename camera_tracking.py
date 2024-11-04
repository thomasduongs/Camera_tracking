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



import tf.transformations
import numpy.linalg as la
import matplotlib.pyplot as plt
from threading import Lock, Thread
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from geometry_msgs.msg import Twist, PoseArray, PoseWithCovarianceStamped, Pose, PoseStamped
from gazebo_msgs.msg import LinkStates
from std_msgs.msg import Float32MultiArray, Int32MultiArray
from cv_bridge import CvBridge, CvBridgeError
from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats


sys.path.insert(1, os.path.abspath("."))
from lib.params import BASE_DEST_TRANSFORM, VISION_IMAGE_TOPIC, REALSENSE2CAMERA, REALSENSE_IMAGE_TOPIC
from lib.board_tracker import BoardTracker
from lib.utils import *
from src.fetch_controller_python.fetch_robot import FetchRobot


class PoseTracker:
    def __init__(self):
        self.cameraPoseTilt = 0
        self.cameraPosePan = 0

bridge = CvBridge()

def execute(tag_pose): 
    dx = tag_pose.position.x
    dy = tag_pose.position.y
    dz = tag_pose.position.z
    # dx, dy, dz = T
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

    print(ptk.cameraPosePan)
    print(ptk.cameraPoseTilt)

    if np.abs(newHorizontal) > 1.5708:
        newHorizontal = ptk.cameraPosePan

    if newVertical > 1.5708 or newVertical < -0.785:
        newVertical = ptk.cameraPoseTilt

    inputMatrix = [[newHorizontal, newVertical], [0.0 for _ in range(2)], [0.0 for _ in range(2)]]
    robot.lookAt(inputMatrix)

    ptk.cameraPosePan = newHorizontal
    ptk.cameraPoseTilt = newVertical




def posefromMatrix(matrix):
    m = matrix[:,:3]

    cur_matrix = m.reshape(3,4)
    cur_matrix_homo = np.vstack((cur_matrix, np.array([0, 0, 0, 1]))) # to homogenous coordinates

    q = tf.transformations.quaternion_from_matrix(cur_matrix_homo)

    p = Pose()
    p.position.x = matrix[0][3]
    p.position.y = matrix[1][3]
    p.position.z = matrix[2][3] 

    roll, pitch, yaw = tf.transformations.euler_from_quaternion(q)



    p.orientation.x = roll
    p.orientation.y = pitch
    p.orientation.z = yaw
    # p.orientation.w = 0.0

    return p

def image_callback(img_msg): 
    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))


    try:
            pose_array = PoseArray()
            pose_array.header.frame_id = "map"
            results = tracker.detect(cv_image)
            # print(results)
            ids = []
            poses = []
            
            for result in results:
                realsense_tag_pose_m, _, _ = tracker.detectPose(result)
                op = MatrixOperation()

                tag_pose_m = np.dot(REALSENSE2CAMERA, realsense_tag_pose_m)

                M = robot.getTransformationBase2Camera()
                pose_in_baseFr = np.dot(M, tag_pose_m)

                # tag_pose_position, tag_pose_roll, tag_pose_pitch, tag_pose_yaw = posefromMatrix(pose_in_baseFr)
                tag_pose = posefromMatrix(pose_in_baseFr)
                execute(tag_pose)



    # try:
    #     gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    #     results = tracker.detect(gray)

    #     M, _, T = tracker.getTransformation(results[0])
    #     execute(T)
    #     publisher.publish(my_msg.reshape(-1))
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


 
