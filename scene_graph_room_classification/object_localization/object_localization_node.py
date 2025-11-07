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
        """
        Initializes the Object3DBoundingBoxNode.

        This node computes 3D bounding boxes for detected objects by combining
        2D segmentation masks with depth information. It also visualizes the robot's
        pose and trajectory for debugging purposes.
        """
        # Initialize the parent Node class with the node name
        super().__init__('object_3d_bbox_node')

        # Create CvBridge object for converting between ROS and OpenCV images
        self.bridge = CvBridge()

        # Initialize storage for camera calibration parameters and robot pose
        self.camera_info = None
        self.latest_odom = None

        # Subscribe to camera calibration info (intrinsic parameters)
        # This is needed to convert 2D pixels to 3D points
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        # Subscribe to robot odometry for pose information
        self.odom_sub = self.create_subscription(Odometry, '/scene_graph/sync/odom', self.odom_callback, 10)

        # Set up synchronized subscribers for RGB image, depth image, and detected objects
        self.color_sub = Subscriber(self, Image, '/scene_graph/sync/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/scene_graph/sync/depth/image_raw')
        self.objects_sub = Subscriber(self, ObjectSegmentList, '/scene_graph/object_segments')

        # Synchronize the three topics with ApproximateTimeSynchronizer
        # queue_size=10: buffer size for message synchronization
        # slop=0.1: maximum 100ms time difference allowed between synchronized messages
        self.sync = ApproximateTimeSynchronizer(
            [self.color_sub, self.depth_sub, self.objects_sub],
            queue_size=10,
            slop=0.1  # 100ms tolerance
        )

        # Register callback function for synchronized messages
        self.sync.registerCallback(self.synchronized_callback)

        # Create publisher for bounding box visualization markers in RViz
        self.bbox_marker_pub = self.create_publisher(
            MarkerArray,
            '/scene_graph/debug/bounding_boxes',
            10
        )

        # Create publisher for robot pose visualization
        self.pose_marker_pub = self.create_publisher(MarkerArray, "/scene_graph/viz/robot_pose", 10)

        # Initialize trajectory tracking variables
        self.trajectory_points = []
        self.max_trajectory_points = 1000  # Limit trajectory history to prevent memory overflow

        # Create publisher for 3D bounding box data (not just visualization)
        self.bbox_3d_pub = self.create_publisher(Object3DBoundingBoxList, '/scene_graph/bounding_boxes_3d', 10)

        self.get_logger().info('Object 3D Bounding Box Node initialized')
        

    def camera_info_callback(self, msg):
        """
        Callback for camera calibration information.

        Stores the camera intrinsic parameters (focal length, principal point)
        needed for pixel-to-3D conversion. Only stores the first message received.

        Args:
            msg (sensor_msgs/CameraInfo): Camera calibration parameters
        """
        if self.camera_info is None:
            self.camera_info = msg
            self.get_logger().info('Camera info received')
            

    def odom_callback(self, msg):
        """
        Callback for robot odometry data.

        Stores the latest robot pose and creates visualization markers showing:
        1. An arrow indicating position and orientation
        2. A sphere at the robot's position
        3. A line strip showing the robot's trajectory

        Args:
            msg (nav_msgs/Odometry): Robot odometry containing pose and velocity
        """
        # Store the latest odometry message
        self.latest_odom = msg

        # Create a marker array to hold multiple visualization markers
        marker_array = MarkerArray()

        # 1. Create arrow marker for robot pose (shows both position and orientation)
        arrow_marker = self.create_pose_arrow(msg)
        marker_array.markers.append(arrow_marker)

        # 2. Create sphere marker for position only (easier to see than arrow tip)
        position_marker = self.create_position_sphere(msg)
        marker_array.markers.append(position_marker)

        # 3. Create and update trajectory line showing robot's path history
        self.trajectory_points.append(msg.pose.pose.position)
        
        # Remove oldest point if trajectory exceeds maximum length
        if len(self.trajectory_points) > self.max_trajectory_points:
            self.trajectory_points.pop(0)
        trajectory_marker = self.create_trajectory(msg.header.frame_id)
        marker_array.markers.append(trajectory_marker)

        # Publish all visualization markers
        self.pose_marker_pub.publish(marker_array)
        

    def create_pose_arrow(self, odom_msg):
        """
        Create an ARROW marker showing robot's position and orientation.

        The arrow points in the direction the robot is facing, providing
        visual feedback on both location and heading.

        Args:
            odom_msg (nav_msgs/Odometry): Odometry message with pose information

        Returns:
            visualization_msgs/Marker: Arrow marker for RViz
        """
        marker = Marker()
        marker.header = odom_msg.header
        
        # FIXME: use position from SLAM instead of raw odometry
        marker.header.frame_id = "map"
        marker.ns = "robot_pose"  # Namespace for grouping markers
        marker.id = 0  # Unique ID within namespace
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Set pose from odometry (position and orientation)
        marker.pose = odom_msg.pose.pose

        # Arrow dimensions (length, width, height)
        marker.scale.x = 0.5   # Arrow length (0.5 meters)
        marker.scale.y = 0.05  # Arrow shaft width
        marker.scale.z = 0.05  # Arrow shaft height

        # Color (Red arrow for visibility)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0  # Fully opaque

        # Persistent marker (lifetime=0 means it stays until deleted)
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()

        return marker
    

    def create_position_sphere(self, odom_msg):
        """
        Create a SPHERE marker at the robot's position.

        This provides an easy-to-see indicator of the robot's current location,
        complementing the arrow marker.

        Args:
            odom_msg (nav_msgs/Odometry): Odometry message with pose information

        Returns:
            visualization_msgs/Marker: Sphere marker for RViz
        """
        marker = Marker()
        marker.header = odom_msg.header
        # FIXME: use position from SLAM instead of raw odometry
        marker.header.frame_id = "map"
        marker.ns = "robot_position"
        marker.id = 1  # Different ID from arrow marker
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Set position only (orientation doesn't matter for a sphere)
        marker.pose.position = odom_msg.pose.pose.position
        marker.pose.orientation.w = 1.0  # Identity quaternion

        # Sphere size (0.2m diameter)
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2

        # Color (Green sphere)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8  # Semi-transparent

        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()

        return marker
    

    def create_trajectory(self, frame_id):
        """
        Create a LINE_STRIP marker showing the robot's path history.

        This visualizes where the robot has been by connecting all previous
        position points with a continuous line.

        Args:
            frame_id (str): Reference frame for the marker

        Returns:
            visualization_msgs/Marker: Line strip marker for RViz
        """
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "trajectory"
        marker.id = 2  # Different ID from pose markers
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Line width
        marker.scale.x = 0.02

        # Color (Blue trajectory)
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.8  # Semi-transparent

        # Add all stored trajectory points to the line
        for point in self.trajectory_points:
            marker.points.append(point)

        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()

        return marker
    

    def pixel_to_3d(self, u, v, depth):
        """
        Convert 2D pixel coordinates to 3D point using camera intrinsics.

        Uses the pinhole camera model to back-project a 2D pixel with depth
        into 3D space. The formula is:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth

        Args:
            u (float): Pixel x-coordinate
            v (float): Pixel y-coordinate
            depth (float): Depth value in meters

        Returns:
            tuple: (X, Y, Z) in camera frame, or None if invalid
        """
        # Validate inputs: check for camera info and valid depth
        if self.camera_info is None or depth <= 0 or np.isnan(depth) or np.isinf(depth):
            return None

        # Extract camera intrinsic parameters from calibration
        fx = self.camera_info.k[0]  # Focal length in x direction
        fy = self.camera_info.k[4]  # Focal length in y direction
        cx = self.camera_info.k[2]  # Principal point x (image center)
        cy = self.camera_info.k[5]  # Principal point y (image center)

        # Convert to 3D coordinates in camera frame using pinhole projection
        Z = depth
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy

        return (X, Y, Z)

    def compute_3d_bounding_box(self, segment_pixels, depth_image):
        """
        Compute 3D bounding box from 2D segment pixels and depth information.

        This function:
        1. Converts each 2D pixel in the segmentation mask to 3D using depth
        2. Applies Statistical Outlier Removal (SOR) to filter noise
        3. Computes an axis-aligned bounding box around the filtered points
        4. Returns the bounding box center, dimensions, and corner points

        Args:
            segment_pixels (list): List of Point32 representing 2D pixel coordinates
            depth_image (numpy.ndarray): Array of depth values

        Returns:
            dict: Contains 'center', 'min_point', 'max_point', 'dimensions', 
                  'corners', and point statistics, or None if invalid
        """
        # Check if camera calibration is available
        if self.camera_info is None:
            self.get_logger().warn('Camera info not available yet')
            return None

        points_3d = []

        # Convert each 2D segment pixel to 3D point using depth
        for pixel in segment_pixels:
            u = int(pixel.x)
            v = int(pixel.y)

            # Check if pixel is within image bounds
            if 0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]:
                depth = depth_image[v, u]

                # Convert depth encoding if necessary
                # Assuming depth is in millimeters (uint16 format is common)
                if depth > 0:
                    depth_meters = depth / 1000.0  # Convert mm to meters
                    point_3d = self.pixel_to_3d(u, v, depth_meters)
                    if point_3d is not None:
                        points_3d.append(point_3d)

        # Need at least 3 points to compute a meaningful bounding box
        if len(points_3d) < 3:
            self.get_logger().warn(f'Not enough valid 3D points: {len(points_3d)}')
            return None

        # Convert to numpy array for vectorized operations
        points_3d = np.array(points_3d)

        # Statistical Outlier Removal (SOR) filter to remove noise points
        k_neighbors = min(10, len(points_3d) - 1)  # Use 10 neighbors or max available
        std_multiplier = 0.1  # Standard deviation multiplier for threshold (conservative)

        if len(points_3d) > k_neighbors:
            from scipy.spatial import KDTree

            # Build KD-tree for efficient nearest neighbor search
            tree = KDTree(points_3d)
            mean_distances = []

            # For each point, compute mean distance to k nearest neighbors
            for point in points_3d:
                # Query k+1 neighbors (includes the point itself at distance 0)
                distances, _ = tree.query(point, k=k_neighbors+1)
                # Exclude distance to itself (first element)
                mean_distances.append(np.mean(distances[1:]))

            mean_distances = np.array(mean_distances)

            # Calculate global statistics of mean neighbor distances
            global_mean = np.mean(mean_distances)
            global_std = np.std(mean_distances)

            # Remove points whose mean neighbor distance exceeds threshold
            # This filters out isolated points that are likely depth sensor noise
            threshold = global_mean + std_multiplier * global_std
            inlier_mask = mean_distances <= threshold
            points_3d_filtered = points_3d[inlier_mask]

            # Verify we still have enough points after filtering
            if len(points_3d_filtered) < 3:
                self.get_logger().warn(f'Not enough points after filtering: {len(points_3d_filtered)}')
                return None
        else:
            # Skip SOR filter if not enough points for k-NN
            points_3d_filtered = points_3d
            self.get_logger().info('Skipping SOR filter: not enough points for k-NN')

        # Compute axis-aligned bounding box (AABB) from filtered points
        min_point = np.min(points_3d_filtered, axis=0)
        max_point = np.max(points_3d_filtered, axis=0)
        center = (min_point + max_point) / 2.0
        dimensions = max_point - min_point

        # Compute all 8 corners of the bounding box
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

        # Return comprehensive bounding box information
        return {
            'center': center,
            'min_point': min_point,
            'max_point': max_point,
            'dimensions': dimensions,
            'corners': np.array(corners),
            'num_points': len(points_3d_filtered),
            'num_points_original': len(points_3d),
            'num_outliers_removed': len(points_3d) - len(points_3d_filtered)
        }
        

    def synchronized_callback(self, color_msg, depth_msg, objects_msg):
        """
        Process synchronized RGB image, depth image, and detected objects.

        This is the main processing callback that:
        1. Converts ROS images to OpenCV format
        2. For each detected object, computes a 3D bounding box
        3. Transforms bounding boxes to the map frame using robot pose
        4. Publishes 3D bounding boxes and visualization markers

        Args:
            color_msg (sensor_msgs/Image): RGB camera image
            depth_msg (sensor_msgs/Image): Depth image
            objects_msg (ObjectSegmentList): List of detected objects with 2D segments
        """
        try:
            # Convert ROS images to OpenCV format
            color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            # Use 'passthrough' to preserve original depth encoding (e.g., 16-bit)
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            self.get_logger().debug(
                f'Processing {len(objects_msg.objects)} objects, '
                f'depth encoding: {depth_msg.encoding}'
            )

            # Create message to hold all 3D bounding boxes
            bbox_list_msg = Object3DBoundingBoxList()
            bbox_list_msg.header.stamp = self.get_clock().now().to_msg()

            # Process each detected object from the segmentation node
            for obj in objects_msg.objects:
                self.get_logger().debug(f'Processing object: {obj.class_name.data}')

                # Compute 3D bounding box from 2D segmentation and depth
                start_time = time.time()
                bbox_3d = self.compute_3d_bounding_box(obj.segment, depth_image)
                end_time = time.time()
                self.get_logger().info(f'Time to compute 3D bounding box: {end_time - start_time}')

                if bbox_3d is not None:
                    # Log detailed bounding box information
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

                    # Create Point32 messages for bounding box center and dimensions
                    bbox_center = Point32()
                    bbox_center.x = float(bbox_3d["center"][0])
                    bbox_center.y = float(bbox_3d["center"][1])
                    bbox_center.z = float(bbox_3d["center"][2])

                    bbox_dimensions = Point32()
                    bbox_dimensions.x = float(bbox_3d["dimensions"][0])
                    bbox_dimensions.y = float(bbox_3d["dimensions"][1])
                    bbox_dimensions.z = float(bbox_3d["dimensions"][2])

                    # Calculate minimum corner point (front-left-bottom)
                    min_point = Point32()
                    min_point.x = bbox_center.x - bbox_dimensions.x / 2.0
                    min_point.y = bbox_center.y - bbox_dimensions.y / 2.0
                    min_point.z = bbox_center.z - bbox_dimensions.z / 2.0

                    # Calculate maximum corner point (back-right-top)
                    max_point = Point32()
                    max_point.x = bbox_center.x + bbox_dimensions.x / 2.0
                    max_point.y = bbox_center.y + bbox_dimensions.y / 2.0
                    max_point.z = bbox_center.z + bbox_dimensions.z / 2.0

                    # Transform bounding box corners from camera frame to map frame
                    # using the robot's current pose from odometry
                    min_point = self.transform_point(min_point, self.latest_odom.pose.pose)
                    max_point = self.transform_point(max_point, self.latest_odom.pose.pose)

                    # Create bounding box message
                    bbox_msg = Object3DBoundingBox()
                    bbox_msg.name = obj.class_name
                    bbox_msg.bounding_box.append(min_point)
                    bbox_msg.bounding_box.append(max_point)

                    bbox_list_msg.bbox.append(bbox_msg)
                else:
                    self.get_logger().warn(
                        f'Could not compute 3D bbox for object: {obj.class_name.data}'
                    )

            # Publish bounding boxes as visualization markers and data messages
            self.publish_boxes(bbox_list_msg)
            self.bbox_3d_pub.publish(bbox_list_msg)

        except Exception as e:
            self.get_logger().error(f'Error in synchronized callback: {str(e)}')
            
            

    def quaternion_to_rotation_matrix(self, q):
        """
        Convert quaternion to 3x3 rotation matrix.

        Uses the standard quaternion-to-matrix conversion formula.
        Quaternions represent rotations in a compact, singularity-free form.

        Args:
            q (geometry_msgs/Quaternion): Quaternion (x, y, z, w)

        Returns:
            numpy.ndarray: 3x3 rotation matrix
        """
        # Normalize quaternion to unit length (required for valid rotation)
        norm = math.sqrt(q.x**2 + q.y**2 + q.z**2 + q.w**2)
        q.x /= norm
        q.y /= norm
        q.z /= norm
        q.w /= norm

        # Convert quaternion to rotation matrix using standard formula
        R = np.array([
            [1 - 2*(q.y**2 + q.z**2), 2*(q.x*q.y - q.w*q.z), 2*(q.x*q.z + q.w*q.y)],
            [2*(q.x*q.y + q.w*q.z), 1 - 2*(q.x**2 + q.z**2), 2*(q.y*q.z - q.w*q.x)],
            [2*(q.x*q.z - q.w*q.y), 2*(q.y*q.z + q.w*q.x), 1 - 2*(q.x**2 + q.y**2)]
        ])

        return R
    

    def transform_point(self, point, odom_pose):
        """
        Transform a Point32 from robot's local camera frame to map frame.

        Applies rotation and translation to convert coordinates from the
        camera's reference frame to the global map frame using the robot's pose.

        Args:
            point (Point32): Point in robot's local camera frame
            odom_pose (geometry_msgs/Pose): Robot pose from odometry

        Returns:
            Point32: Transformed point in map frame
        """
        # FIXME: These manual angle adjustments suggest a frame transform issue
        # Should use proper TF2 transforms instead
        odom_pose.orientation.z = odom_pose.orientation.z - np.pi / 2
        odom_pose.orientation.y = odom_pose.orientation.y + np.pi / 2

        # Get rotation matrix from quaternion
        R = self.quaternion_to_rotation_matrix(odom_pose.orientation)

        # Convert point to numpy array for matrix operations
        p_local = np.array([point.x, point.y, point.z])

        # Apply rotation: p_rotated = R * p_local
        p_rotated = R @ p_local

        # Apply translation: p_global = p_rotated + robot_position
        p_global = p_rotated + np.array([
            odom_pose.position.x,
            odom_pose.position.y,
            odom_pose.position.z
        ])

        # Create transformed point message
        transformed = Point32()
        transformed.x = float(p_global[0])
        transformed.y = float(p_global[1])
        transformed.z = float(p_global[2])

        return transformed
    

    def create_bbox_marker(self, min_pt, max_pt, marker_id, frame_id='map',
                          color=None, namespace='bounding_boxes'):
        """
        Create a CUBE marker for visualizing a 3D bounding box in RViz.

        Args:
            min_pt (Point32): Minimum corner of bounding box
            max_pt (Point32): Maximum corner of bounding box
            marker_id (int): Unique ID for this marker
            frame_id (str): Reference frame (default: 'map')
            color (tuple): RGBA color values 0-1 (default: red)
            namespace (str): Marker namespace for grouping

        Returns:
            visualization_msgs/Marker: Cube marker for RViz
        """
        # Extract coordinates from Point32 messages
        x_min, y_min, z_min = min_pt.x, min_pt.y, min_pt.z
        x_max, y_max, z_max = max_pt.x, max_pt.y, max_pt.z

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.CUBE  # Use cube for solid bounding box visualization

        # Set pose to center of bounding box
        marker.pose.position.x = (min_pt.x + max_pt.x) / 2
        marker.pose.position.y = (min_pt.y + max_pt.y) / 2  # Note: potential typo in original (uses z twice)
        marker.pose.position.z = (min_pt.z + max_pt.z) / 2

        # Set scale to dimensions of bounding box
        marker.scale.x = max_pt.x - min_pt.x
        marker.scale.y = max_pt.y - min_pt.y
        marker.scale.z = max_pt.z - min_pt.z

        # Set color (default to red if not specified)
        if color is None:
            color = (1.0, 0.0, 0.0, 1.0)
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = 0.3  # Semi-transparent for better visibility

        return marker
    

    def publish_boxes(self, bbox_msg):
        """
        Publish visualization markers for all bounding boxes.

        Creates RViz markers for each bounding box in the message and
        publishes them as a MarkerArray for visualization.

        Args:
            bbox_msg (Object3DBoundingBoxList): List of 3D bounding boxes
        """
        marker_array = MarkerArray()

        # Create a visualization marker for each bounding box
        for i, bbox in enumerate(bbox_msg.bbox):
            marker = self.create_bbox_marker(
                bbox.bounding_box[0],  # Min point
                bbox.bounding_box[1],  # Max point
                marker_id=i,
                frame_id='map',
                color=(1.0, 0.0, 0.0, 1.0)  # Red boxes
            )
            marker_array.markers.append(marker)

        # Publish all markers at once
        self.bbox_marker_pub.publish(marker_array)
        

def main(args=None):
    """
    Main entry point for the ROS2 node.

    Initializes ROS2, creates the Object3DBoundingBoxNode, and enters
    the spin loop to process callbacks until shutdown.

    Args:
        args: Command line arguments (optional)
    """
    # Initialize the ROS2 Python client library
    rclpy.init(args=args)

    # Create an instance of the Object3DBoundingBoxNode
    node = Object3DBoundingBoxNode()

    try:
        # Enter the ROS2 event loop to process callbacks
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Allow graceful shutdown on Ctrl+C
        pass
    finally:
        # Clean up the node and shutdown ROS2
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    # Execute main function when script is run directly
    main()
