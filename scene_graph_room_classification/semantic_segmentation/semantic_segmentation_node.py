#!/usr/bin/env python3

from ultralytics import YOLO

import rclpy
from rclpy.node import Node

import torch

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time
from scene_graph_interfaces.msg import DetectedObjects, DetectedObject
from geometry_msgs.msg import Point32
from std_msgs.msg import String
from nav_msgs.msg import Odometry

import message_filters


class SemanticSegmentationNode(Node):
    def __init__(self):
        super().__init__('yolov9_seg_node')
        
        # Load YOLOv9 model
        self.model = YOLO('yolov9e-seg.pt')
        # self.model.to('cuda')
        # self.model.eval()
        
        self.n = 0
        self.in_cb = False
        self.rgb_image = Image()
        self.depth_msg = PointCloud2()
        self.odom_msg = Odometry()
        
        # Create a CvBridge object for converting ROS images to OpenCV
        self.bridge = CvBridge()
        
        # Subscribe to the camera image topic
        self.image_sub = message_filters.Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = message_filters.Subscriber(self, PointCloud2, '/camera/depth/points')
        self.odom_sub = message_filters.Subscriber(self, Odometry, '/odom')
        
        # Approximate Time Synchronizer allows slight time differences between topics
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub, self.odom_sub], 
            queue_size=100, 
            slop=0.1
        )
        self.ts.registerCallback(self.synchronized_callback)
        
        # Publisher for the segmented image
        self.image_pub = self.create_publisher(Image, '/scene_graph/color/image_raw', 1)
        self.segmented_image_pub = self.create_publisher(Image, '/camera/color/segmented_image', 1)
        self.depth_pub = self.create_publisher(PointCloud2, '/scene_graph/depth/points', 1)
        self.odom_pub = self.create_publisher(Odometry, '/scene_graph/odom', 1)
        self.detected_objects_pub = self.create_publisher(DetectedObjects, '/scene_graph/detected_objects', 10)
        
        self.get_logger().info('YOLOv9 Segmentation Node initialized')

    def synchronized_callback(self, ros_image, depth_msg, odom_msg):
        # Convert ROS Image message to OpenCV format
        print('in synchronized callback')
        start_time = time.time()
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge Error: {e}")
            return
        
        self.rgb_image = ros_image
        
        # Perform segmentation using YOLOv9 model
        results = self.model(cv_image)
        
        # Extract segmentation masks and apply them to the original image
        masks = results[0].masks.data if results[0].masks is not None else None
        
        if masks is None:
            self.get_logger().warn("No masks found in the image.")
        
        detected_objects = []
        indices = []
        
        for result in results:
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls.item())
                class_name = result.names[class_id]
                confidence = box.conf.item()
                
                if confidence > 0.6:
                    x1, y1, x2, y2 = box.xyxy[0][0].item(), box.xyxy[0][1].item(), box.xyxy[0][2].item(), box.xyxy[0][3].item()
                    print(f"Class: {class_name}, Confidence: {confidence}, Coordinates: ({x1}, {y1}), ({x2}, {y2})")
                    
                    detected_objects.append(DetectedObject(
                        String(data=str(class_name)), 
                        [Point32(x=x1, y=y1, z=0.0), Point32(x=x2, y=y2, z=0.0)], 
                        []
                    ))
                    indices.append(i)
        
        h2, w2, _ = results[0].orig_img.shape
        masks = None
        
        # Define range of brightness in HSV
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([0, 0, 1])
        
        self.get_logger().info(f'[TIMING]: {time.time() - start_time}')
        
        n = 0
        for i in indices:
            mask = results[0].masks[i]
            segment = []
            
            for point in mask.xy[0]:
                segment.append(Point32(x=int(point[0]), y=int(point[1]), z=0.0))
            
            detected_objects[n].segment = segment
            n += 1
            
            mask_raw = mask.cpu().data.numpy().transpose(1, 2, 0)
            
            # Convert single channel grayscale to 3 channel image
            mask_3channel = cv2.cvtColor(mask_raw, cv2.COLOR_GRAY2BGR)
            
            # Resize the mask to the same size as the image
            mask = cv2.resize(mask_3channel, (w2, h2))
            
            # Create a mask. Threshold the HSV image to get everything black
            mask = cv2.inRange(mask, lower_black, upper_black)
            
            # Invert the mask to get everything but black
            mask = cv2.bitwise_not(mask)
            
            if masks is None:
                masks = mask
            else:
                masks = cv2.bitwise_or(mask, masks)
        
        # Apply the mask to the original image
        masked = cv2.bitwise_and(results[0].orig_img, results[0].orig_img, mask=masks)
        self.segmented_image_pub.publish(self.bridge.cv2_to_imgmsg(masked, encoding='bgr8'))
        
        detected_objects_msg = DetectedObjects()
        detected_objects_msg.objects = detected_objects
        detected_objects_msg.header.stamp = self.get_clock().now().to_msg()
        
        depth_msg.header.stamp = self.get_clock().now().to_msg()
        self.rgb_image.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        
        self.rgb_image.header.frame_id = 'map'
        depth_msg.header.frame_id = 'map'
        odom_msg.header.frame_id = 'map'
        
        print('depth: ', depth_msg.header.stamp.nanosec)
        print('image: ', self.rgb_image.header.stamp.nanosec)
        print('odom: ', odom_msg.header.stamp.nanosec)
        print('detected objects: ', detected_objects_msg.header.stamp.nanosec)
        
        # Publish depth image and odometry together with segmented image for synchronization
        self.image_pub.publish(self.rgb_image)
        self.depth_pub.publish(depth_msg)
        self.odom_pub.publish(odom_msg)
        self.detected_objects_pub.publish(detected_objects_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SemanticSegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
