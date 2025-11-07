#!/usr/bin/env python

from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from scene_graph_interfaces.msg import ObjectSegmentList, ObjectSegment
from geometry_msgs.msg import Point32
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import message_filters

class ImageSegmentationNode(Node):
    def __init__(self):
        """
        Initializes the ImageSegmentationNode.

        This constructor sets up the ROS2 node for performing image segmentation using YOLO.
        It subscribes to camera image, depth, and odometry topics, and publishes synchronized 
        topics along with detected object information.
        """
        # Initialize the parent Node class with the node name
        super().__init__('image_segmentation_node')

        # Load the YOLO segmentation model (YOLOv11x-seg variant)
        self.model = YOLO('yolo11x-seg.pt')
        # Move the model to GPU for faster inference
        self.model.to('cuda')

        # Set confidence threshold - only detections above 0.8 confidence are kept
        self.confidence_threshold = 0.8

        # Initialize counter and callback flag (currently unused in the code)
        self.n = 0
        self.in_cb = False

        # Initialize message storage variables for latest sensor data
        self.rgb_image = Image()
        self.depth_msg = PointCloud2()
        self.odom_msg = Odometry()

        # Create a CvBridge object for converting between ROS Image messages and OpenCV images
        self.bridge = CvBridge()

        # Subscribe to the RGB camera image topic
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        # Subscribe to the depth image topic
        self.depth_image_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        # Subscribe to the odometry topic for robot pose information
        self.odom_sub = message_filters.Subscriber(self, Odometry, '/odom')

        # Create an ApproximateTimeSynchronizer to handle messages that arrive at slightly different times
        # queue_size=100: buffer size for incoming messages
        # slop=0.1: maximum time difference (in seconds) allowed between synchronized messages
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_image_sub, self.odom_sub],
            queue_size=100,
            slop=0.1
        )

        # Register the callback function to be called when synchronized messages arrive
        self.ts.registerCallback(self.synchronized_callback)

        # Create publishers for synchronized sensor topics (queue_size=1 for latest data only)
        self.image_pub = self.create_publisher(Image, '/scene_graph/sync/color/image_raw', 1)
        self.depth_image_pub = self.create_publisher(Image, '/scene_graph/sync/depth/image_raw', 1)
        self.odom_pub = self.create_publisher(Odometry, '/scene_graph/sync/odom', 1)

        # Create publisher for debugging - shows the segmented image with masks applied
        self.segmented_image_pub = self.create_publisher(Image, '/scene_graph/debug/segmented_image', 1)

        # Create publisher for detected objects with segmentation information (queue_size=10)
        self.detected_objects_pub = self.create_publisher(ObjectSegmentList, '/scene_graph/object_segments', 10)
        

    def synchronized_callback(self, ros_image, depth_msg, odom_msg):
        """
        Callback function for synchronized sensor data.

        This function is triggered when synchronized RGB image, depth image, and odometry 
        messages are available. It performs object detection and segmentation using YOLO,
        extracts masks and bounding boxes, and publishes the results.

        Args:
            ros_image (sensor_msgs/Image): RGB camera image
            depth_msg (sensor_msgs/Image): Depth image (note: parameter doc says PointCloud2 but it's Image)
            odom_msg (nav_msgs/Odometry): Robot odometry/pose information
        """
        # Convert ROS Image message to OpenCV format (BGR color space)
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        except CvBridgeError as e:
            # Log error and exit callback if conversion fails
            self.get_logger().error(f"CvBridge Error: {e}")
            return

        # Store the RGB image for later publishing
        self.rgb_image = ros_image

        # Run YOLO inference on the image to detect and segment objects
        results = self.model(cv_image)

        # Extract segmentation masks from results (None if no masks detected)
        masks = results[0].masks.data if results[0].masks is not None else None

        # Warn if no masks were detected
        if masks is None:
            self.get_logger().warn("No masks found in the image.")

        # Initialize lists to store detected objects and their indices
        detected_objects = []
        indices = []

        # Iterate through all detection results
        for result in results:
            for i, box in enumerate(result.boxes):
                # Extract class ID and name for the detected object
                class_id = int(box.cls.item())
                class_name = result.names[class_id]

                # Get confidence score for this detection
                confidence = box.conf.item()

                # Only process detections above the confidence threshold
                if confidence > self.confidence_threshold:
                    # Extract bounding box coordinates (x1, y1) = top-left, (x2, y2) = bottom-right
                    x1, y1, x2, y2 = box.xyxy[0][0].item(), box.xyxy[0][1].item(), box.xyxy[0][2].item(), box.xyxy[0][3].item()

                    # Log detection information for debugging
                    self.get_logger().debug(f"Class: {class_name}, Confidence: {confidence}, Coordinates: ({x1}, {y1}), ({x2}, {y2})")

                    # Create ObjectSegment message with class name and bounding box
                    # Segment points will be filled in later
                    detected_objects.append(ObjectSegment(
                        class_name=String(data=str(class_name)),
                        bounding_box=[Point32(x=x1, y=y1, z=0.0), Point32(x=x2, y=y2, z=0.0)],
                        segment=[]
                    ))

                    # Store the index for later mask processing
                    indices.append(i)

        # Get dimensions of the original image for mask resizing
        h2, w2, _ = results[0].orig_img.shape

        # Reset masks variable to build combined mask
        masks = None

        # Define color range for identifying black pixels (for mask inversion)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([0, 0, 1])

        # Counter for detected objects
        n = 0

        # Process each detected object's segmentation mask
        for i in indices:
            # Get the mask for this specific detection
            mask = results[0].masks[i]

            # Extract polygon points defining the segmentation boundary
            segment = []
            for point in mask.xy[0]:
                # Convert each point to Point32 message format
                segment.append(Point32(x=float(point[0]), y=float(point[1]), z=0.0))

            # Add segment points to the corresponding detected object
            detected_objects[n].segment = segment
            n += 1

            # Process the mask for visualization
            # Move mask from GPU to CPU and convert to numpy array
            mask_raw = mask.cpu().data.numpy().transpose(1, 2, 0)

            # Convert grayscale mask to 3-channel BGR format
            mask_3channel = cv2.cvtColor(mask_raw, cv2.COLOR_GRAY2BGR)

            # Resize mask to match original image dimensions
            mask = cv2.resize(mask_3channel, (w2, h2))

            # Create binary mask by identifying non-black pixels
            mask = cv2.inRange(mask, lower_black, upper_black)

            # Invert the mask (swap black and white regions)
            mask = cv2.bitwise_not(mask)

            # Combine with existing masks using OR operation
            if masks is None:
                masks = mask
            else:
                masks = cv2.bitwise_or(mask, masks)

        # Apply the combined mask to the original image to show only segmented regions
        masked = cv2.bitwise_and(results[0].orig_img, results[0].orig_img, mask=masks)

        # Publish the masked image for debugging/visualization
        self.segmented_image_pub.publish(self.bridge.cv2_to_imgmsg(masked, encoding='bgr8'))

        # Create ObjectSegmentList message containing all detected objects
        detected_objects_msg = ObjectSegmentList()
        detected_objects_msg.objects = detected_objects

        # Update timestamps for all messages to the current time
        detected_objects_msg.header.stamp = self.get_clock().now().to_msg()
        depth_msg.header.stamp = self.get_clock().now().to_msg()
        self.rgb_image.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.stamp = self.get_clock().now().to_msg()

        # Set frame_id to 'map' for all messages (coordinate frame reference)
        depth_msg.header.frame_id = 'map'
        self.rgb_image.header.frame_id = 'map'
        odom_msg.header.frame_id = 'map'

        # Publish all synchronized messages and detected objects
        self.image_pub.publish(self.rgb_image)
        self.depth_image_pub.publish(depth_msg)
        self.odom_pub.publish(odom_msg)
        self.detected_objects_pub.publish(detected_objects_msg)
        

def main(args=None):
    """
    Main entry point for the ROS2 node.

    Initializes the ROS2 system, creates the ImageSegmentationNode,
    and enters the spin loop to process callbacks.

    Args:
        args: Command line arguments (optional)
    """
    # Initialize the ROS2 Python client library
    rclpy.init(args=args)

    # Create an instance of the ImageSegmentationNode
    node = ImageSegmentationNode()

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
