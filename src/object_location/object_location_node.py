#!/usr/bin/env python3

import rospy
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point32
import sensor_msgs.point_cloud2 as pc2
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from scipy.spatial.transform import Rotation as R
import message_filters
from shapely.geometry import Point, Polygon
from scene_graph.msg import DetectedObjects, GraphObjects, GraphObject
import time


# Global publishers
debug_image_pub = None
debug_depth_pub = None
debug_odom_pub = None
graph_objects_pub = None


def convert_to_global_point(pose, point):
    """
    Convert a local point to global frame using robot's pose
    
    Args:
        pose: Odometry message containing robot pose
        point: Point32 in local frame
    
    Returns:
        Point32 in global frame
    """
    local_point = np.array([point.x, point.y, point.z])
    
    # Extract quaternion from pose
    q = [pose.pose.pose.orientation.x,
         pose.pose.pose.orientation.y,
         pose.pose.pose.orientation.z,
         pose.pose.pose.orientation.w]
    
    # Create rotation matrix from quaternion
    rotation = R.from_quat(q)
    rotation_matrix = rotation.as_matrix()
    
    # Rotate the local point by the robot's orientation
    rotated_point = rotation_matrix @ local_point
    
    # Translate the rotated point by the robot's global position
    global_point = Point32()
    global_point.x = rotated_point[0] + pose.pose.pose.position.x
    global_point.y = rotated_point[1] + pose.pose.pose.position.y
    global_point.z = rotated_point[2] + pose.pose.pose.position.z
    
    return global_point


def rotate_point(point, roll, pitch, yaw):
    """
    Rotate a point by given roll, pitch, yaw angles
    
    Args:
        point: Point32 to rotate
        roll: rotation around x-axis in radians
        pitch: rotation around y-axis in radians
        yaw: rotation around z-axis in radians
    
    Returns:
        Rotated Point32
    """
    # Create rotation from euler angles
    rotation = R.from_euler('xyz', [roll, pitch, yaw])
    rotation_matrix = rotation.as_matrix()
    
    # Apply rotation
    point_array = np.array([point.x, point.y, point.z])
    rotated = rotation_matrix @ point_array
    
    result = Point32()
    result.x = rotated[0]
    result.y = rotated[1]
    result.z = rotated[2]
    
    return result


def compute_median(values):
    """Compute median of a list of values"""
    if len(values) == 0:
        raise ValueError("Cannot compute median of an empty list")
    
    return np.median(values)


def compute_median_point(points):
    """
    Compute median point from a list of Point32 objects
    
    Args:
        points: list of Point32 objects
    
    Returns:
        Point32 representing median point
    """
    if len(points) == 0:
        raise ValueError("The input list of points is empty")
    
    x_values = [p.x for p in points]
    y_values = [p.y for p in points]
    z_values = [p.z for p in points]
    
    median_point = Point32()
    median_point.x = compute_median(x_values)
    median_point.y = compute_median(y_values)
    median_point.z = compute_median(z_values)
    
    return median_point


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two Point32 objects"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)


def is_point_in_polygon(polygon_points, px, py):
    """
    Check if a point is inside a polygon using Shapely
    
    Args:
        polygon_points: list of Point32 defining polygon vertices
        px, py: coordinates of point to check
    
    Returns:
        True if point is inside or on boundary of polygon
    """
    if len(polygon_points) < 3:
        return False
    
    # Create polygon from points
    polygon_coords = [(p.x, p.y) for p in polygon_points]
    polygon = Polygon(polygon_coords)
    point = Point(px, py)
    
    # Check if point is inside or on boundary
    return polygon.contains(point) or polygon.boundary.contains(point)


def get_nearest_points(points, center):
    """
    Get the nearest 70% of points to a center point
    
    Args:
        points: list of Point32 objects
        center: Point32 center point
    
    Returns:
        list of nearest Point32 objects (70% closest)
    """
    # Calculate distances
    distances = [(calculate_distance(p, center), p) for p in points]
    
    # Sort by distance
    distances.sort(key=lambda x: x[0])
    
    # Get closest 70%
    num_points_to_select = int(len(distances) * 0.70)
    nearest_points = [p for _, p in distances[:num_points_to_select]]
    
    return nearest_points


def compute_bounding_box(points):
    """
    Compute axis-aligned bounding box from points
    
    Args:
        points: list of Point32 objects
    
    Returns:
        tuple of (min_point, max_point) as Point32 objects
    """
    if len(points) == 0:
        raise ValueError("The input list of points is empty")
    
    min_point = Point32()
    max_point = Point32()
    
    x_values = [p.x for p in points]
    y_values = [p.y for p in points]
    z_values = [p.z for p in points]
    
    min_point.x = min(x_values)
    min_point.y = min(y_values)
    min_point.z = min(z_values)
    
    max_point.x = max(x_values)
    max_point.y = max(y_values)
    max_point.z = max(z_values)
    
    return (min_point, max_point)


def rotate_and_transform_points_vectorized(points_array, pose):
    """
    Vectorized version of point cloud transformation
    
    Args:
        points_array: Nx3 numpy array of points
        pose: Odometry message containing robot pose
    
    Returns:
        Nx3 numpy array of transformed points
    """
    # Apply y-axis offset (vectorized)
    points_array[:, 1] -= 0.65
    
    # First rotation: around z-axis by -pi/2
    rot1 = R.from_euler('z', -np.pi/2)
    points_array = rot1.apply(points_array)
    
    # Second rotation: around y-axis by pi/2
    rot2 = R.from_euler('y', np.pi/2)
    points_array = rot2.apply(points_array)
    
    # Convert to global frame
    # Extract quaternion from pose
    q = [pose.pose.pose.orientation.x,
         pose.pose.pose.orientation.y,
         pose.pose.pose.orientation.z,
         pose.pose.pose.orientation.w]
    
    # Create rotation matrix from quaternion
    rotation = R.from_quat(q)
    
    # Apply rotation and translation (vectorized)
    points_array = rotation.apply(points_array)
    points_array[:, 0] += pose.pose.pose.position.x
    points_array[:, 1] += pose.pose.pose.position.y
    points_array[:, 2] += pose.pose.pose.position.z
    
    return points_array


def synchronized_callback(image, depth_cloud, pose, detected_objects):
    """
    Callback for synchronized sensor data
    
    Args:
        image: sensor_msgs/Image
        depth_cloud: sensor_msgs/PointCloud2
        pose: nav_msgs/Odometry
        detected_objects: scene_graph/DetectedObjects
    """
    global debug_image_pub, debug_depth_pub, debug_odom_pub, graph_objects_pub
    
    rospy.loginfo("Received synchronized messages")
    
    # Read point cloud data
    rospy.loginfo("Converting to global frame...")
    start = time.time()

    # Convert point cloud to numpy array in one operation
    points_array = np.array(list(pc2.read_points(depth_cloud, skip_nans=False, field_names=("x", "y", "z"))))

    # Apply all transformations at once using vectorized operations
    points_transformed = rotate_and_transform_points_vectorized(points_array, pose)

    # Create output point cloud
    header = depth_cloud.header
    cloud_out = pc2.create_cloud_xyz32(header, points_transformed.tolist())

    # Publish debug topics
    debug_image_pub.publish(image)
    debug_depth_pub.publish(cloud_out)
    debug_odom_pub.publish(pose)

    end = time.time()
    print(f"Processing time: {end - start:.3f} seconds")
    
    # Process detected objects
    rospy.loginfo("Processing detected objects...")
    graph_objects = GraphObjects()
    graph_objects.header.stamp = rospy.Time.now()
    
    for detected_object in detected_objects.objects:
        print(f"Detected object: {detected_object.class_name}")
        
        pointcloud_segment = []
        
        # Extract bounding box coordinates
        bbox_y_min = int(detected_object.bounding_box[0].y)
        bbox_y_max = int(detected_object.bounding_box[1].y)
        bbox_x_min = int(detected_object.bounding_box[0].x)
        bbox_x_max = int(detected_object.bounding_box[1].x)
        
        # Iterate through bounding box pixels
        for y in range(bbox_y_min, bbox_y_max):
            for x in range(bbox_x_min, bbox_x_max):
                # Check bounds
                if y < 0 or y >= image.height or x < 0 or x >= image.width:
                    continue
                
                # Check if point is in segmentation polygon
                if not is_point_in_polygon(detected_object.segment, x, y):
                    continue
                
                array_position = y * image.width + x
                
                # Check array bounds
                if array_position < 0 or array_position >= len(points_transformed):
                    rospy.logwarn(f"array_position out of bounds: {array_position} max: {len(points_transformed)}")
                    continue
                
                # Get transformed point
                px, py, pz = points_transformed[array_position]
                
                # Only add points with positive z
                if pz > 0.0:
                    p = Point32()
                    p.x = px
                    p.y = py
                    p.z = pz
                    pointcloud_segment.append(p)
        
        if len(pointcloud_segment) == 0:
            continue
        
        # Compute center and filter points
        center_point = compute_median_point(pointcloud_segment)
        filtered_pointcloud_segment = get_nearest_points(pointcloud_segment, center_point)
        
        if len(filtered_pointcloud_segment) == 0:
            continue
        
        # Compute bounding box
        bounding_box = compute_bounding_box(filtered_pointcloud_segment)
        
        # Create graph object
        graph_object = GraphObject()
        graph_object.name = detected_object.class_name
        graph_object.bounding_box.append(bounding_box[0])
        graph_object.bounding_box.append(bounding_box[1])
        
        graph_objects.objects.append(graph_object)
    
    rospy.loginfo("Publishing graph objects...")
    graph_objects_pub.publish(graph_objects)


def main():
    """Main function to initialize node and start processing"""
    global debug_image_pub, debug_depth_pub, debug_odom_pub, graph_objects_pub
    
    # Initialize ROS node
    rospy.init_node('synchronizer_node', anonymous=True)
    
    # Create publishers
    debug_image_pub = rospy.Publisher('/scene_graph/debug/image', Image, queue_size=10)
    debug_depth_pub = rospy.Publisher('/scene_graph/debug/depth', PointCloud2, queue_size=10)
    debug_odom_pub = rospy.Publisher('/scene_graph/debug/odom', Odometry, queue_size=10)
    graph_objects_pub = rospy.Publisher('/scene_graph/seen_graph_objects', GraphObjects, queue_size=100)
    
    # Create subscribers with message filters
    image_sub = message_filters.Subscriber('/scene_graph/color/image_raw', Image)
    pointcloud_sub = message_filters.Subscriber('/scene_graph/depth/points', PointCloud2)
    pose_sub = message_filters.Subscriber('/scene_graph/odom', Odometry)
    detected_objects_sub = message_filters.Subscriber('/scene_graph/detected_objects', DetectedObjects)
    
    # Create approximate time synchronizer
    ts = message_filters.ApproximateTimeSynchronizer(
        [image_sub, pointcloud_sub, pose_sub, detected_objects_sub],
        queue_size=10,
        slop=0.1
    )
    
    ts.registerCallback(synchronized_callback)
    
    rospy.loginfo("Synchronizer node started")
    rospy.spin()


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
