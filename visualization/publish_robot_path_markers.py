#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Quaternion
import tf.transformations as tf_trans
import math

class RobotPathPublisher:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('robot_path_publisher')

        # Subscriber to the /odom topic
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # Publishers for markers
        self.path_marker_pub = rospy.Publisher('/visualization_marker_path', Marker, queue_size=10)
        self.current_marker_pub = rospy.Publisher('/visualization_marker_position', Marker, queue_size=10)
        self.arrow_marker_pub = rospy.Publisher('/visualization_marker_orientation', Marker, queue_size=10)

        # Initialize the marker for the robot's path
        self.path_marker = Marker()
        self.path_marker.header.frame_id = "map"  # Reference frame is map
        self.path_marker.type = Marker.LINE_STRIP  # Marker type as line strip to connect points
        self.path_marker.action = Marker.ADD  # Action type
        self.path_marker.scale.x = 0.05  # Thickness of the line
        self.path_marker.color.r = 0.0  # Color of the path (green)
        self.path_marker.color.g = 0.45
        self.path_marker.color.b = 0.81
        self.path_marker.color.a = 1.0  # Alpha (opacity)

        # Initialize the marker for the robot's current position
        self.current_position_marker = Marker()
        self.current_position_marker.header.frame_id = "map"  # Reference frame is map
        self.current_position_marker.type = Marker.SPHERE  # Marker type as a sphere
        # self.current_position_marker.type = Marker.MESH_RESOURCE
        # self.current_position_marker.mesh_resource = "package://scene_graph/meshes/turtlebot.dae"
        self.current_position_marker.action = Marker.ADD  # Action type
        self.current_position_marker.scale.x = 0.4  # Size of the sphere
        self.current_position_marker.scale.y = 0.4
        self.current_position_marker.scale.z = 0.4
        self.current_position_marker.color.r = 0.0  # Color of the current position (blue)
        self.current_position_marker.color.g = 0.45
        self.current_position_marker.color.b = 0.81
        self.current_position_marker.color.a = 1.0  # Alpha (opacity)

        # Initialize the marker for the robot's orientation (arrow)
        self.orientation_marker = Marker()
        self.orientation_marker.header.frame_id = "map"  # Reference frame is map
        self.orientation_marker.type = Marker.ARROW  # Marker type as an arrow
        self.orientation_marker.action = Marker.ADD  # Action type
        self.orientation_marker.scale.x = 0.07  # Length of the arrow
        self.orientation_marker.scale.y = 0.1  # Arrow width
        self.orientation_marker.scale.z = 0.1  # Arrow height
        self.orientation_marker.color.r = 1.0  # Color of the arrow (red)
        self.orientation_marker.color.g = 0.0
        self.orientation_marker.color.b = 0.0
        self.orientation_marker.color.a = 1.0  # Alpha (opacity)

    def odom_callback(self, msg):
        # Extract the current position and orientation from the odometry message
        current_position = msg.pose.pose.position
        current_orientation = msg.pose.pose.orientation

        # Create a Point message for the current position (for the path)
        path_point = Point()
        path_point.x = current_position.x
        path_point.y = current_position.y
        path_point.z = current_position.z

        # Add the current position as a point in the path marker
        self.path_marker.points.append(path_point)

        # Update the current position marker with the latest position
        self.current_position_marker.pose.position = current_position

        # Calculate the arrow direction using the robot's orientation (quaternion)
        arrow_start = Point()
        arrow_start.x = current_position.x
        arrow_start.y = current_position.y
        arrow_start.z = current_position.z

        # Convert quaternion to Euler angles to get the robot's yaw (rotation around z-axis)
        orientation_q = [current_orientation.x, current_orientation.y, current_orientation.z, current_orientation.w]
        (roll, pitch, yaw) = tf_trans.euler_from_quaternion(orientation_q)

        # Define the arrow endpoint (use yaw to determine direction)
        arrow_end = Point()
        arrow_length = 0.5  # Length of the arrow
        arrow_end.x = arrow_start.x + arrow_length * math.cos(yaw)
        arrow_end.y = arrow_start.y + arrow_length * math.sin(yaw)
        arrow_end.z = arrow_start.z

        # Set arrow start and end positions
        self.orientation_marker.points = [arrow_start, arrow_end]

        # Set the timestamp and frame ID for all markers
        self.path_marker.header.stamp = rospy.Time.now()
        self.current_position_marker.header.stamp = rospy.Time.now()
        self.orientation_marker.header.stamp = rospy.Time.now()

        # Publish the markers
        self.path_marker_pub.publish(self.path_marker)
        self.current_marker_pub.publish(self.current_position_marker)
        self.arrow_marker_pub.publish(self.orientation_marker)

    def run(self):
        # Keep the node running
        rospy.spin()

if __name__ == '__main__':
    try:
        # Create an instance of the RobotPathPublisher class and run it
        node = RobotPathPublisher()
        node.run()
    except rospy.ROSInterruptException:
        pass
