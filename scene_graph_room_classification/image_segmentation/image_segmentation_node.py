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
        Initializes the SemanticSegmentationNode.
        
        Subscribes to the camera image, depth and odometry topics and 
        publishes the synchronized topics, as well as the detected objects.
        """
        
        super().__init__('image_segmentation_node')
        
        # Load YOLO model
        self.model = YOLO('yolo11x-seg.pt')
        self.model.to('cuda')
        self.confidence_threshold = 0.8
        
        self.n = 0
        self.in_cb = False
        self.rgb_image = Image()
        self.depth_msg = PointCloud2()
        self.odom_msg = Odometry()
        
        # Create a CvBridge object for converting ROS images to OpenCV
        self.bridge = CvBridge()
        
        # Subscribe to the camera image, depth and odometry topic
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_image_sub = message_filters.Subscriber(self, Image, '/camera/depth/image_raw')
        self.odom_sub = message_filters.Subscriber(self, Odometry, '/odom')
        
        # Approximate Time Synchronizer allows slight time differences between topics
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_image_sub, self.odom_sub], 
            queue_size=100, 
            slop=0.1
        )
        self.ts.registerCallback(self.synchronized_callback)
        
        # Create publishers for synchronized topics, as well as for debugging and the detected objects
        self.image_pub = self.create_publisher(Image, '/scene_graph/color/image_raw', 1)
        self.depth_image_pub = self.create_publisher(Image, '/scene_graph/depth/image_raw', 1)
        self.segmented_image_pub = self.create_publisher(Image, '/scene_graph/debug/segmented_image', 1)
        self.odom_pub = self.create_publisher(Odometry, '/scene_graph/odom', 1)
        self.detected_objects_pub = self.create_publisher(ObjectSegmentList, '/scene_graph/object_segments', 10)


    def synchronized_callback(self, ros_image, depth_msg, odom_msg):
        """
        Callback for synchronized sensor data
        
        Args:
            ros_image: sensor_msgs/Image
            depth_msg: sensor_msgs/PointCloud2
            odom_msg: nav_msgs/Odometry
        
        This function takes in synchronized RGB and depth images, as well as odometry data.
        It converts the ROS Image message to OpenCV format, applies model inference
        to detect objects in the image, extracts the segmentation masks and applies them to the
        original image. The function then creates a DetectedObjects message and publishes it
        together with the synchronized RGB and depth images and odometry data.
        """
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return
        
        self.rgb_image = ros_image
        
        # Model inference
        results = self.model(cv_image)
        
        # Extract segmentation masks and apply them to the original image
        masks = results[0].masks.data if results[0].masks is not None else None
        
        if masks is None:
            self.get_logger().warn("No masks found in the image.")
        
        detected_objects = []
        indices = []
        
        # Extract detected objects
        for result in results:
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls.item())
                class_name = result.names[class_id]
                confidence = box.conf.item()
                
                if confidence > self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0][0].item(), box.xyxy[0][1].item(), box.xyxy[0][2].item(), box.xyxy[0][3].item()
                    self.get_logger().debug(f"Class: {class_name}, Confidence: {confidence}, Coordinates: ({x1}, {y1}), ({x2}, {y2})")
                    
                    detected_objects.append(ObjectSegment(
                        class_name=String(data=str(class_name)), 
                        bounding_box=[Point32(x=x1, y=y1, z=0.0), Point32(x=x2, y=y2, z=0.0)], 
                        segment=[]
                    ))
                    indices.append(i)
        
        # Create masks for debugging
        h2, w2, _ = results[0].orig_img.shape
        masks = None
        
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([0, 0, 1])
        
        n = 0
        for i in indices:
            mask = results[0].masks[i]
            segment = []
            
            for point in mask.xy[0]:
                segment.append(Point32(x=float(point[0]), y=float(point[1]), z=0.0))
            
            detected_objects[n].segment = segment
            n += 1
            
            mask_raw = mask.cpu().data.numpy().transpose(1, 2, 0)
            mask_3channel = cv2.cvtColor(mask_raw, cv2.COLOR_GRAY2BGR)
            mask = cv2.resize(mask_3channel, (w2, h2))
            mask = cv2.inRange(mask, lower_black, upper_black)
            mask = cv2.bitwise_not(mask)
            
            if masks is None:
                masks = mask
            else:
                masks = cv2.bitwise_or(mask, masks)
        
        # Apply the mask to the original image
        masked = cv2.bitwise_and(results[0].orig_img, results[0].orig_img, mask=masks)
        self.segmented_image_pub.publish(self.bridge.cv2_to_imgmsg(masked, encoding='bgr8'))
        
        # Create DetectedObjects message
        detected_objects_msg = ObjectSegmentList()
        detected_objects_msg.objects = detected_objects
        detected_objects_msg.header.stamp = self.get_clock().now().to_msg()
        
        depth_msg.header.stamp = self.get_clock().now().to_msg()
        self.rgb_image.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        
        depth_msg.header.frame_id = 'map'
        self.rgb_image.header.frame_id = 'map'
        odom_msg.header.frame_id = 'map'
        
        # Publish depth image and odometry together with segmented image for synchronization
        self.image_pub.publish(self.rgb_image)
        self.depth_image_pub.publish(depth_msg)
        self.odom_pub.publish(odom_msg)
        self.detected_objects_pub.publish(detected_objects_msg)


def main(args=None):
    """
    Main entry point for the node. Initializes the ROS node and
    starts the spin loop.
    """
    
    rclpy.init(args=args)
    node = ImageSegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
