#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point32
from cv_bridge import CvBridge
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from scene_graph_interfaces.msg import Object3DBoundingBox, Object3DBoundingBoxList, ObjectSegmentList
from visualization_msgs.msg import Marker, MarkerArray
import time
import math


class Object3DBoundingBoxNode(Node):
    def __init__(self):
        super().__init__('object_3d_bbox_node')

        self.bridge = CvBridge()
        self.camera_info = None
        self.latest_odom = None

        # Subscribe to camera info (only need one since they're synchronized)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        # Subscribe to odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Set up synchronized subscribers for images and detected objects
        self.color_sub = Subscriber(self, Image, '/scene_graph/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/scene_graph/depth/image_raw')
        self.objects_sub = Subscriber(self, ObjectSegmentList, '/scene_graph/object_segments')

        # Synchronize the three topics
        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.objects_sub],
            queue_size=10,
            slop=0.1  # 100ms tolerance
        )
        self.sync.registerCallback(self.synchronized_callback)
        
        # Publisher for bounding box marker array
        self.bbox_marker_pub = self.create_publisher(
            MarkerArray, 
            '/scene_graph/debug/bounding_boxes', 
            10
        )
        
        self.pose_marker_pub = self.create_publisher(MarkerArray, "/scene_graph/viz/robot_pose", 10)
        self.trajectory_points = []
        self.max_trajectory_points = 1000  # Limit trajectory length

        # Publisher for 3D bounding boxes
        self.bbox_3d_pub = self.create_publisher(Object3DBoundingBoxList, '/scene_graph/bounding_boxes_3d', 10)

        self.get_logger().info('Object 3D Bounding Box Node initialized')


    def camera_info_callback(self, msg):
        """Store camera intrinsics"""
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info('Camera info received')


    def odom_callback(self, msg):
        """Store latest odometry"""
        self.latest_odom = msg
        
        marker_array = MarkerArray()
        
        # 1. Create arrow marker for robot pose (position + orientation)
        arrow_marker = self.create_pose_arrow(msg)
        marker_array.markers.append(arrow_marker)
        
        # 2. Create sphere marker for position only
        position_marker = self.create_position_sphere(msg)
        marker_array.markers.append(position_marker)
        
        # 3. Create trajectory line (path history)
        self.trajectory_points.append(msg.pose.pose.position)
        if len(self.trajectory_points) > self.max_trajectory_points:
            self.trajectory_points.pop(0)
        
        trajectory_marker = self.create_trajectory(msg.header.frame_id)
        marker_array.markers.append(trajectory_marker)
        
        # Publish all markers
        self.pose_marker_pub.publish(marker_array)
        
        
    def create_pose_arrow(self, odom_msg):
        """Create an ARROW marker showing position and orientation."""
        marker = Marker()
        marker.header = odom_msg.header
        
        # FIXME: use position from SLAM
        marker.header.frame_id = "map"
        
        marker.ns = "robot_pose"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Set pose from odometry
        marker.pose = odom_msg.pose.pose
        
        # Arrow dimensions (length, width, height)
        marker.scale.x = 0.5  # Arrow length
        marker.scale.y = 0.05  # Arrow width
        marker.scale.z = 0.05  # Arrow height
        
        # Color (Red arrow)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()  # Persistent
        
        return marker
    
    def create_position_sphere(self, odom_msg):
        """Create a SPHERE marker at robot's position."""
        marker = Marker()
        marker.header = odom_msg.header
        
        # FIXME: use position from SLAM
        marker.header.frame_id = "map"
        
        marker.ns = "robot_position"
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Set position
        marker.pose.position = odom_msg.pose.pose.position
        marker.pose.orientation.w = 1.0
        
        # Sphere size
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        
        # Color (Green sphere)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        
        return marker
    
    def create_trajectory(self, frame_id):
        """Create a LINE_STRIP marker showing the robot's path."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trajectory"
        marker.id = 2
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Line width
        marker.scale.x = 0.02
        
        # Color (Blue trajectory)
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.8
        
        # Add all trajectory points
        for point in self.trajectory_points:
            marker.points.append(point)
        
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        
        return marker


    def pixel_to_3d(self, u, v, depth):
        """
        Convert 2D pixel coordinates to 3D point using camera intrinsics

        Args:
            u, v: pixel coordinates (x, y)
            depth: depth value in meters

        Returns:
            tuple: (X, Y, Z) in camera frame, or None if invalid
        """
        if self.camera_info is None or depth <= 0 or np.isnan(depth) or np.isinf(depth):
            return None

        # Extract intrinsics from camera_info
        fx = self.camera_info.k[0]  # Focal length x
        fy = self.camera_info.k[4]  # Focal length y
        cx = self.camera_info.k[2]  # Principal point x
        cy = self.camera_info.k[5]  # Principal point y

        # Convert to 3D coordinates in camera frame
        Z = depth
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        return (X, Y, Z)


    def compute_3d_bounding_box(self, segment_pixels, depth_image):
        """
        Compute 3D bounding box from 2D segment pixels and depth information

        Args:
            segment_pixels: list of Point32 representing 2D pixel coordinates
            depth_image: numpy array of depth values

        Returns:
            dict with 'center', 'min_point', 'max_point', 'corners' or None
        """
        if self.camera_info is None:
            self.get_logger().warn('Camera info not available yet')
            return None

        points_3d = []

        # Convert each segment pixel to 3D
        for pixel in segment_pixels:
            u = int(pixel.x)
            v = int(pixel.y)

            # Check bounds
            if 0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]:
                depth = depth_image[v, u]

                # Convert depth encoding if necessary
                # Assuming depth is in millimeters (uint16)
                if depth > 0:
                    depth_meters = depth / 1000.0  # Convert mm to meters

                    point_3d = self.pixel_to_3d(u, v, depth_meters)
                    if point_3d is not None:
                        points_3d.append(point_3d)

        if len(points_3d) < 3:
            self.get_logger().warn(f'Not enough valid 3D points: {len(points_3d)}')
            return None

        # Convert to numpy array for easier computation
        points_3d = np.array(points_3d)

        # Compute axis-aligned bounding box
        min_point = np.min(points_3d, axis=0)
        max_point = np.max(points_3d, axis=0)
        center = (min_point + max_point) / 2.0
        dimensions = max_point - min_point

        # Compute 8 corners of the bounding box
        corners = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    corner = [
                        min_point[0] if i == 0 else max_point[0],
                        min_point[1] if j == 0 else max_point[1],
                        min_point[2] if k == 0 else max_point[2]
                    ]
                    corners.append(corner)

        return {
            'center': center,
            'min_point': min_point,
            'max_point': max_point,
            'dimensions': dimensions,
            'corners': np.array(corners),
            'num_points': len(points_3d)
        }


    def synchronized_callback(self, color_msg, depth_msg, objects_msg):
        """
        Process synchronized messages to create 3D bounding boxes
        """
        try:
            # Convert ROS images to OpenCV format
            color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')

            # Depth image - use 'passthrough' to preserve original encoding
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            self.get_logger().debug(
                f'Processing {len(objects_msg.objects)} objects, '
                f'depth encoding: {depth_msg.encoding}'
            )
            
            bbox_list_msg = Object3DBoundingBoxList()
            bbox_list_msg.header.stamp = self.get_clock().now().to_msg()

            # Process each detected object
            for obj in objects_msg.objects:
                self.get_logger().debug(f'Processing object: {obj.class_name.data}')

                # Compute 3D bounding box from segment
                start_time = time.time()
                bbox_3d = self.compute_3d_bounding_box(obj.segment, depth_image)
                end_time = time.time()
                self.get_logger().info(f'Time to compute 3D bounding box: {end_time - start_time}')

                if bbox_3d is not None:
                    self.get_logger().info(
                        f'Object "{obj.class_name.data}" - '
                        f'Center: [{bbox_3d["center"][0]:.2f}, '
                        f'{bbox_3d["center"][1]:.2f}, '
                        f'{bbox_3d["center"][2]:.2f}], '
                        f'Dimensions: [{bbox_3d["dimensions"][0]:.2f}, '
                        f'{bbox_3d["dimensions"][1]:.2f}, '
                        f'{bbox_3d["dimensions"][2]:.2f}] m, '
                        f'Points: {bbox_3d["num_points"]}'
                    )
                    
                    bbox_center = Point32()
                    bbox_center.x = float(bbox_3d["center"][0])
                    bbox_center.y = float(bbox_3d["center"][1])
                    bbox_center.z = float(bbox_3d["center"][2])
                    
                    bbox_dimensions = Point32()
                    bbox_dimensions.x = float(bbox_3d["dimensions"][0])
                    bbox_dimensions.y = float(bbox_3d["dimensions"][1])
                    bbox_dimensions.z = float(bbox_3d["dimensions"][2])
                    
                    # Calculate min point
                    min_point = Point32()
                    min_point.x = bbox_center.x - bbox_dimensions.x / 2.0
                    min_point.y = bbox_center.y - bbox_dimensions.y / 2.0
                    min_point.z = bbox_center.z - bbox_dimensions.z / 2.0

                    # Calculate max point
                    max_point = Point32()
                    max_point.x = bbox_center.x + bbox_dimensions.x / 2.0
                    max_point.y = bbox_center.y + bbox_dimensions.y / 2.0
                    max_point.z = bbox_center.z + bbox_dimensions.z / 2.0
                    
                    # Transfom min and max points using the robot's pose
                    min_point = self.transform_point(min_point, self.latest_odom.pose.pose)
                    max_point = self.transform_point(max_point, self.latest_odom.pose.pose)


                    bbox_msg = Object3DBoundingBox()
                    bbox_msg.name = obj.class_name
                    bbox_msg.bounding_box.append(min_point)
                    bbox_msg.bounding_box.append(max_point)
                    
                    bbox_list_msg.bbox.append(bbox_msg)
                    
                    
                else:
                    self.get_logger().warn(
                        f'Could not compute 3D bbox for object: {obj.class_name.data}'
                    )
                    
            self.publish_boxes(bbox_list_msg)
                    
            self.bbox_3d_pub.publish(bbox_list_msg)

        except Exception as e:
            self.get_logger().error(f'Error in synchronized callback: {str(e)}')
            
    def quaternion_to_rotation_matrix(self, q):
        """
        Convert quaternion to 3x3 rotation matrix.
        
        Args:
            q: geometry_msgs/Quaternion
        Returns:
            3x3 numpy array rotation matrix
        """
        # Normalize quaternion
        norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        q.x /= norm
        q.y /= norm
        q.z /= norm
        q.w /= norm
        
        # Quaternion to rotation matrix
        R = np.array([
            [1 - 2*(q.y**2 + q.z**2),     2*(q.x*q.y - q.w*q.z),     2*(q.x*q.z + q.w*q.y)],
            [    2*(q.x*q.y + q.w*q.z), 1 - 2*(q.x**2 + q.z**2),     2*(q.y*q.z - q.w*q.x)],
            [    2*(q.x*q.z - q.w*q.y),     2*(q.y*q.z + q.w*q.x), 1 - 2*(q.x**2 + q.y**2)]
        ])
        
        return R
    
    def transform_point(self, point, odom_pose):
        """
        Transform a Point32 from robot's local frame to odom frame.
        
        Args:
            point: Point32 in robot's local frame
            odom_pose: Pose from odometry message
        Returns:
            Point32 in odom frame
        """
        
        # FIXME: why is this necessary?
        odom_pose.orientation.z = odom_pose.orientation.z - np.pi / 2
        odom_pose.orientation.y = odom_pose.orientation.y + np.pi / 2
        
        # Get rotation matrix from quaternion
        R = self.quaternion_to_rotation_matrix(odom_pose.orientation)
        
        # Point as numpy array
        p_local = np.array([point.x, point.y, point.z])
        
        # Apply rotation
        p_rotated = R @ p_local
        
        # Apply translation
        p_global = p_rotated + np.array([
            odom_pose.position.x,
            odom_pose.position.y,
            odom_pose.position.z
        ])
        
        # Create transformed point
        transformed = Point32()
        transformed.x = float(p_global[0])
        transformed.y = float(p_global[1])
        transformed.z = float(p_global[2])
        
        return transformed
         
            
    def create_bbox_marker(self, min_pt, max_pt, marker_id, frame_id='map', 
                          color=None, namespace='bounding_boxes'):
        """
        Create a LINE_LIST marker for a bounding box.
        
        Args:
            min_pt: tuple (x, y, z) - minimum corner
            max_pt: tuple (x, y, z) - maximum corner
            marker_id: unique ID for this marker
            frame_id: reference frame
            color: tuple (r, g, b, a) - color values 0-1
            namespace: marker namespace
        """
        
        # Extract coordinates from Point32
        x_min, y_min, z_min = min_pt.x, min_pt.y, min_pt.z
        x_max, y_max, z_max = max_pt.x, max_pt.y, max_pt.z
        
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.pose.position.x = (min_pt.x + max_pt.x) / 2
        marker.pose.position.y = (min_pt.z + max_pt.y) / 2
        marker.pose.position.z = (min_pt.z + max_pt.z) / 2
        marker.scale.x = max_pt.x - min_pt.x
        marker.scale.y = max_pt.y - min_pt.y
        marker.scale.z = max_pt.z - min_pt.z
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.3  # Semi-transparent
        
        return marker
        
    
    def publish_boxes(self, bbox_msg):
        
        marker_array = MarkerArray()     
           
        for i, (bbox) in enumerate(bbox_msg.bbox):
            marker = self.create_bbox_marker(
                bbox.bounding_box[0], bbox.bounding_box[1], 
                marker_id=i,
                frame_id='map',
                color=(1.0, 0.0, 0.0, 1.0)
            )
            marker_array.markers.append(marker)
        
        self.bbox_marker_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = Object3DBoundingBoxNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
