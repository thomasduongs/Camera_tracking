import rospy 
import sys, time, os
sys.path.insert(1, os.path.abspath("."))
from lib.params import BASE_DEST_TRANSFORM, VISION_IMAGE_TOPIC, REALSENSE2CAMERA, REALSENSE_IMAGE_TOPIC
from lib.board_tracker import BoardTracker
from lib.utils import *
from src.fetch_controller_python.fetch_robot import FetchRobot
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



class PoseTracker:
    def __init__(self):
        self.cameraPoseTilt = 0
        self.cameraPosePan = 0

bridge = CvBridge()

def execute(tag_pose): 
    joint_states = get_latest_joint_state()
    dx = tag_pose.position.x #forward
    dy = tag_pose.position.y #left and right
    dz = tag_pose.position.z-joint_states[0]-1.15 #up and down
    #Camera at ~ z=-0.95m
    # tilt_joint ranges [-0.785 (U), 1.5708 (D) rad] = [-45, 90] 
    # pan_joint ranges [-1.5708 (R), 1.5708 (L) rad] = [-90, 90] 

    tiltAngle = -np.arctan2(dz, dx)
    panAngle = np.arctan2(dy, dx)

    inputMatrix = [[panAngle, tiltAngle], [0.0 for _ in range(2)], [0.0 for _ in range(2)]]
    robot.lookAt(inputMatrix)

    ptk.cameraPosePan = panAngle
    ptk.cameraPoseTilt = tiltAngle

# Joint and link parameters
MAX_JOINT_VEL = np.array([0.1, 1.25, 1.45, 1.57, 1.52, 1.57, 2.26, 2.26])
JOINT_ACTION_SERVER = 'arm_with_torso_controller/follow_joint_trajectory'
JOINT_NAMES = names = ['torso_lift_joint', 'shoulder_pan_joint', 'shoulder_lift_joint', 
               'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 
               'wrist_flex_joint', 'wrist_roll_joint']
_joint_states = dict()
link_name_left = "fetch::l_gripper_finger_link"
link_name_right = "fetch::r_gripper_finger_link"
base_link_pose = None
link_pose_left = None
link_pose_right = None
# listener = tf.TransformListener()

lock = Lock()

def callback(msg):  
    global link_pose_right
    global link_pose_left
    global base_link_pose
    lock.acquire()

    for i, name in enumerate(msg.name):
        if i >= len(msg.position):
            continue
        _joint_states[name] = msg.position[i]
        
        try:
            tf_matrices = get_tf_matrices(listener)

            matrix = tf_matrices["baseLinkToRightFinger"]
            # print(matrix)

            link_pose_right = posefromMatrix(matrix)

            matrix1 = tf_matrices["baseLinkToLeftFinger"]

            link_pose_left = posefromMatrix(matrix)

        except KeyError:
            pass

    lock.release()

def get_latest_joint_state():
    """
    Returns: A list of the joint values. Values may be None if we do not
        have a value for that joint yet.
    """
    lock.acquire()

    ret = None
    if all(name in _joint_states for name in names):
        ret = [_joint_states[name] for name in names]
    lock.release()
    return ret if ret else None


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
    rospy.Subscriber("/joint_states", JointState, callback)

    while not rospy.is_shutdown():
        rospy.spin()


 
