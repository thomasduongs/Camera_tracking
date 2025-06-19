
#!/usr/bin/env python3
import rospy
import actionlib
import math
import csv
import sys
import time
import os
import numpy as np
import cv2
import tf
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
# from scipy.spatial.transform import Rotation as R

sys.path.insert(1, os.path.abspath("."))
from lib.params import BASE_DEST_TRANSFORM, VISION_IMAGE_TOPIC, REALSENSE2CAMERA, REALSENSE_IMAGE_TOPIC
from lib.board_tracker import BoardTracker
from lib.utils import *
from src.fetch_controller_python.fetch_robot import FetchRobot

# Global variables
bridge = CvBridge()
message_counter = 0


q_matrix = np.array([[1, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1.5, 0, 0],
                     [0, 0, 0, 0, 1.5, 0],
                     [0, 0, 0, 0, 0, 1.5]])

scalar1 = 350

# LQR parameters
lqr_Q = scalar1 * q_matrix

# Define the matrix
r_matrix = np.array([
    [0.9, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0.7, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 10, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 10, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 10]
])

# Define the scalar
scalar = 300

# Multiply the matrix by the scalar
lqr_R = scalar * r_matrix

show_animation = True

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
    return p

def image_callback(img_msg):

    global message_counter
    message_counter += 1

    if message_counter % 1 != 0:
        return

    try:
        cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: {0}".format(e))
        return

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

            print(tag_pose)

            starting_joint_states = get_latest_joint_state()   



            if starting_joint_states is None:
                rospy.logwarn("Joint states are not available")
                continue

            x_compensation = 0.34
            y_compensation = 0
            z_compensation = 0.0
            end_effector_pose = np.array([[link_pose_right.position.x + x_compensation],
                            [link_pose_right.position.y - y_compensation],
                            [link_pose_right.position.z - z_compensation],
                            [link_pose_right.orientation.x],
                            [link_pose_right.orientation.y],
                            [link_pose_right.orientation.z]])

            tag_pose.orientation.x = 0.0
            tag_pose.orientation.y = 0.0
            tag_pose.orientation.z = 0.0

            ustar = lqr_speed_steering_control(end_effector_pose, tag_pose, base_heading_angle)

            ustar = np.insert(ustar, 2, 0, axis=0)
            arm_velocity = ustar[2:, 0].reshape(1, -1)
            base_velocity = ustar[:2, 0].reshape(1, -1)

            clamped_arm_velocity = np.clip(arm_velocity, -MAX_JOINT_VEL, MAX_JOINT_VEL)

            waypoints = [starting_joint_states]

            waypoint = waypoints + (clamped_arm_velocity * dt)

            execute_waypoints_trajectory(waypoint, optimal_dts, arm_velocity)

            publish_base_velocity(base_velocity, duration)

            poses.append(tag_pose)
            ids.append(result.tag_id)

        pose_array.poses = poses
        publisher.publish(pose_array)



        id_array = Int32MultiArray()
        id_array.data = ids
        id_publisher.publish(id_array)
    except Exception as e:
        rospy.logerr("Error in image_callback: {0}".format(e))


def callback(msg):  
    global link_pose_right
    global link_pose_left
    global base_heading_angle
    global base_link_pose
    lock.acquire()

    for i, name in enumerate(msg.name):
        if i >= len(msg.position):
            continue
        _joint_states[name] = msg.position[i]
        
        try:
            tf_matrices = get_tf_matrices(listener)

            matrix = tf_matrices["baseLinkToRightFinger"]

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



def get_tf_matrices(listener):
        transformations = {}
        try:
          (trans,rot) = listener.lookupTransform('/base_link', '/base_link', rospy.Time(0))
          transformations["baseLinkTobaseLink"] = tf.TransformerROS.fromTranslationRotation(listener,translation=trans, rotation=rot)

          (trans,rot) = listener.lookupTransform('/base_link', '/shoulder_pan_link', rospy.Time(0))
          transformations["baseLinkToShoulderPan"] = tf.TransformerROS.fromTranslationRotation(listener,translation=trans, rotation=rot)
          (trans,rot) = listener.lookupTransform('/base_link', '/shoulder_lift_link', rospy.Time(0))
          transformations["baseLinkToShoulderLift"] = tf.TransformerROS.fromTranslationRotation(listener,trans, rot)
          (trans,rot) = listener.lookupTransform('/base_link', '/upperarm_roll_link', rospy.Time(0))
          transformations["baseLinkToUpperarmRoll"] = tf.TransformerROS.fromTranslationRotation(listener,trans, rot)
          (trans,rot) = listener.lookupTransform('/base_link', '/elbow_flex_link', rospy.Time(0))
          transformations["baseLinkToElbowFlex"] = tf.TransformerROS.fromTranslationRotation(listener,trans, rot)
          (trans,rot) = listener.lookupTransform('/base_link', '/forearm_roll_link', rospy.Time(0))
          transformations["baseLinkToForearmRoll"] = tf.TransformerROS.fromTranslationRotation(listener,trans, rot)
          (trans,rot) = listener.lookupTransform('/base_link', '/wrist_flex_link', rospy.Time(0))
          transformations["baseLinkToWristFlex"] = tf.TransformerROS.fromTranslationRotation(listener,trans, rot)
          (trans,rot) = listener.lookupTransform('/base_link', '/wrist_roll_link', rospy.Time(0))
          transformations["baseLinkToWristRoll"] = tf.TransformerROS.fromTranslationRotation(listener,trans, rot)
          (trans,rot) = listener.lookupTransform('/base_link', '/l_gripper_finger_link', rospy.Time(0))
          transformations["baseLinkToLeftFinger"] = tf.TransformerROS.fromTranslationRotation(listener,trans, rot)
          (trans,rot) = listener.lookupTransform('/base_link', '/r_gripper_finger_link', rospy.Time(0))
          transformations["baseLinkToRightFinger"] = tf.TransformerROS.fromTranslationRotation(listener,trans, rot)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            pass
        return transformations

def lqr_speed_steering_control(end_effector_pose, tag_pose, base_heading_angle):

    A = np.eye(6)
    B =  get_B(base_heading_angle)   
    ustar = dlqr(A, B, lqr_Q, lqr_R, tag_pose, end_effector_pose)
    return ustar

def get_B (base_heading_angle):
    global link_pose_right
    
    tf_matrices = get_tf_matrices(listener)
    link_00 = tf_matrices["baseLinkTobaseLink"]
    z31_00 = link_00[[0, 1, 2], 2].reshape([3, ])
    t31_00 = link_00[[0, 1, 2], 3].reshape([3, ])

    link_01 = tf_matrices["baseLinkToShoulderPan"]
    z31_01 = link_01[[0, 1, 2], 2].reshape([3, ])
    t31_01 = link_01[[0, 1, 2], 3].reshape([3, ])

    link_02 = tf_matrices["baseLinkToShoulderLift"]
    z31_02 = link_02[[0, 1, 2], 2].reshape([3, ])
    t31_02 = link_02[[0, 1, 2], 3].reshape([3, ])

    link_03 = tf_matrices["baseLinkToUpperarmRoll"]
    z31_03 = link_03[[0, 1, 2], 2].reshape([3, ])
    t31_03 = link_03[[0, 1, 2], 3].reshape([3, ])

    link_04 = tf_matrices["baseLinkToElbowFlex"]
    z31_04 = link_04[[0, 1, 2], 2].reshape([3, ])
    t31_04 = link_04[[0, 1, 2], 3].reshape([3, ])

    link_05 = tf_matrices["baseLinkToForearmRoll"]
    z31_05 = link_05[[0, 1, 2], 2].reshape([3, ])
    t31_05 = link_05[[0, 1, 2], 3].reshape([3, ])

    link_06 = tf_matrices["baseLinkToWristFlex"]
    z31_06 = link_06[[0, 1, 2], 2].reshape([3, ])
    t31_06 = link_06[[0, 1, 2], 3].reshape([3, ])

    link_07 = tf_matrices["baseLinkToWristRoll"]
    z31_07 = link_07[[0, 1, 2], 2].reshape([3, ])
    t31_07 = link_07[[0, 1, 2], 3].reshape([3, ])
    
    link_08 = tf_matrices["baseLinkToRightFinger"]
    z31_08 = link_08[[0, 1, 2], 2].reshape([3, ])
    t31_08 = link_08[[0, 1, 2], 3].reshape([3, ])

    jacobian = np.zeros([6, 8])

    jacobian[0:3, 0] = np.cross(z31_00, (t31_08 - t31_00))
    jacobian[3:6, 0] = z31_00

    jacobian[0:3, 1] = np.cross(z31_01, (t31_08 - t31_01))
    jacobian[3:6, 1] = z31_01

    jacobian[0:3, 2] = np.cross(z31_02, (t31_08 - t31_02))
    jacobian[3:6, 2] = z31_02

    jacobian[0:3, 3] = np.cross(z31_03, (t31_08 - t31_03))
    jacobian[3:6, 3] = z31_03

    jacobian[0:3, 4] = np.cross(z31_04, (t31_08 - t31_04))
    jacobian[3:6, 4] = z31_04
    
    jacobian[0:3, 5] = np.cross(z31_05, (t31_08 - t31_05))
    jacobian[3:6, 5] = z31_05

    jacobian[0:3, 6] = np.cross(z31_06, (t31_08 - t31_06))
    jacobian[3:6, 6] = z31_06

    jacobian[0:3, 7] = np.cross(z31_07, (t31_08 - t31_07))
    jacobian[3:6, 7] = z31_07

    
    B = np.zeros((6,9))
    B[:, 1:9] = jacobian * dt

    base_heading_angle = 0.0

    B[0,0] = dt * math.cos(base_heading_angle) 

    B[1,0] = dt * math.sin(base_heading_angle)  
    
    return B
        
def dlqr(A, B, Q, R, tag_pose, end_effector_pose):
 
    P, p, rt_c, rt_p = solve_dare(B, Q, R, tag_pose)

    M = la.inv(R + (B.T @ P @ B)) @ B.T

    tag_pose_ = np.array([[tag_pose.position.x],
                               [tag_pose.position.y],
                               [tag_pose.position.z],
                               [tag_pose.orientation.x],
                               [tag_pose.orientation.y],
                               [tag_pose.orientation.z]])
    
    state_error_world = end_effector_pose - (tag_pose_)

    ustar = - M @ (P @ (state_error_world + (rt_c.reshape(-1,1) -  rt_p.reshape(-1,1))) + p )
  

        
    return ustar
   

def solve_dare(B, Q, R, tag_pose):
    P = Q
    P_next = Q

    p = np.array([[0], 
        [0],
        [0],
        [0],
        [0],
        [0]])
    p_next = np.array([[0], 
        [ 0],
        [0],
        [0],
        [0],
        [0]])
    
    horizon = 1

        
    for j in range(horizon-1,-1,-1): 
        
        M = la.inv(R + (B.T @ P_next @ B)) @ B.T
        P_plus = P_next.copy()
        P_next = P_next - P_next @ B @ M @ P_next + Q      

        body_xyz_in_horizon_1 =([tag_pose.position.x], 
                                 [tag_pose.position.y], 
                                 [tag_pose.position.z] )  
        
        body_xyz_in_horizon_2 = ([tag_pose.position.x],
                                  [tag_pose.position.y],
                                  [tag_pose.position.z] )         

        body_xyz_in_horizon_1_orientation = np.array([
            [0.0],
            [0.0],
            [0.0]
        ])

        body_xyz_in_horizon_1 = np.vstack((body_xyz_in_horizon_1, body_xyz_in_horizon_1_orientation))

        body_xyz_in_horizon_2_orientation = np.array([
            [0.0],
            [0.0],
            [0.0]
        ])


        body_xyz_in_horizon_2 = np.vstack((body_xyz_in_horizon_2, body_xyz_in_horizon_2_orientation))

        
        p_plus = p_next.copy()
        p_next = p_next  + P_next @ (body_xyz_in_horizon_1 - body_xyz_in_horizon_2) 
        p_next = p_next - P_next @ B @ M @ P_next @ (body_xyz_in_horizon_1 - body_xyz_in_horizon_2) 
        p_next = p_next  - P_next @ B @ M @ p_next
 
    return P_plus, p_plus, body_xyz_in_horizon_1, body_xyz_in_horizon_2





def execute_waypoints_trajectory(waypoints, t, velocities):

    goal = FollowJointTrajectoryGoal()
    goal.trajectory.joint_names.extend(JOINT_NAMES)


    point = JointTrajectoryPoint()
    goal.trajectory.points.append(point)
    goal.trajectory.points[0].time_from_start = rospy.Duration(t[0])
    for j, p in enumerate(waypoints[0]):
        goal.trajectory.points[0].positions.append(waypoints[0][j])

    waypoints = np.array(waypoints)

    _joint_client.send_goal(goal)


def publish_base_velocity(base_velocities, durations):
    velocity_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    twist = Twist()

    start_time = rospy.Time.now()
    current_duration = 0

    for velocity, duration in zip(base_velocities, durations):
        twist.linear.x = velocity[0]
        twist.angular.z = velocity[1]
        while current_duration < duration:
            velocity_publisher.publish(twist)
            # rate.sleep()
            current_duration = (rospy.Time.now() - start_time).to_sec()
        current_duration = 0

if __name__ == '__main__':
    rospy.init_node('fetch_robot_controller', anonymous=True)

    dt = 0.2
    robot = FetchRobot()
    tracker = BoardTracker()

    listener = tf.TransformListener()

    sub_image = rospy.Subscriber(VISION_IMAGE_TOPIC, Image, image_callback)
    rospy.Subscriber("/joint_states", JointState, callback)

    publisher = rospy.Publisher('tagPoses', PoseArray, queue_size=10)
    id_publisher = rospy.Publisher('tagID', Int32MultiArray, queue_size=10)

    global _joint_client 
    _joint_client = actionlib.SimpleActionClient(
        JOINT_ACTION_SERVER, FollowJointTrajectoryAction)
    _joint_client.wait_for_server(timeout=rospy.Duration(3000))

    base_heading_angle = None

    rospy.spin()



   
