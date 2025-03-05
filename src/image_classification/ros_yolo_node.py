#!/usr/bin/env python3

from ultralytics import YOLO
import rospy
import torch
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time
from scene_graph.msg import DetectedObjects, DetectedObject
from geometry_msgs.msg import Point32
from std_msgs.msg import String
from nav_msgs.msg import Odometry

#!/usr/bin/env python

import rospy
import message_filters
from sensor_msgs.msg import Image, PointCloud2

class YOLOv9SegNode:
    
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('yolov9_seg_node', anonymous=True)

        # Load YOLOv9 model
        self.model = YOLO('yolov9e-seg.pt')
        self.model.to('cuda')
        # self.model.eval()
        self.n = 0
        self.in_cb = False
        
        self.rgb_image = Image()
        self.depth_msg = PointCloud2()
        self.odom_msg = Odometry()

        # Create a CvBridge object for converting ROS images to OpenCV
        self.bridge = CvBridge()

        # Subscribe to the camera image topic
        self.image_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/depth/points', PointCloud2)
        self.odom_sub = message_filters.Subscriber('/odom', Odometry)
        
        # Approximate Time Synchronizer allows slight time differences between topics
        ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub, self.odom_sub], queue_size=100, slop=0.03)
        ts.registerCallback(self.synchronized_callback)

        # Publisher for the segmented image
        self.image_pub = rospy.Publisher('/scene_graph/color/image_raw', Image, queue_size=1)
        self.segmented_image_pub = rospy.Publisher('/camera/color/segmented_image', Image, queue_size=1)
        self.depth_pub = rospy.Publisher('/scene_graph/depth/points', PointCloud2, queue_size=1)
        self.odom_pub = rospy.Publisher('/scene_graph/odom', Odometry, queue_size=1)
        
        self.detected_objects_pub = rospy.Publisher('/scene_graph/detected_objects', DetectedObjects, queue_size=10)
        
        
    def synchronized_callback(self, ros_image, depth_msg, odom_msg):
        # Convert ROS Image message to OpenCV format
        
        # depth_msg = self.depth_msg
        # odom_msg = self.odom_msg
        
        # self.n += 1
        
        # if not self.n % 10 == 0:
        #     return
        

        
        
        print('in synchronized callback')
        
        start_time = time.time()
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        self.rgb_image = ros_image

        # Perform segmentation using YOLOv9 model
        results = self.model(cv_image)
        
        # Extract segmentation masks and apply them to the original image
        masks = results[0].masks.data if results[0].masks is not None else None  # Assuming the output format includes masks in xyn format
        if masks is None:
            rospy.logwarn("No masks found in the image.")
        

        detected_objects = []
        indices = []
                        
        for result in results:
            for i, box in enumerate(result.boxes):
                class_id = int(box.cls.item())
                class_name = result.names[class_id]
                confidence = box.conf.item()
                
                if confidence > 0.7:
                    x1, y1, x2, y2 = box.xyxy[0][0].item(), box.xyxy[0][1].item(), box.xyxy[0][2].item(), box.xyxy[0][3].item()
                    print(f"Class: {class_name}, Confidence: {confidence}, Coordinates: ({x1}, {y1}), ({x2}, {y2})")
                    
                    detected_objects.append(DetectedObject(String(str(class_name)), [Point32(x1, y1, 0.0), Point32(x2, y2, 0.0)], []))
                    indices.append(i)
                    
        
        h2, w2, _ = results[0].orig_img.shape
        masks = None

        # Define range of brightness in HSV
        lower_black = np.array([0,0,0])
        upper_black = np.array([0,0,1])
        
        # print(time.time() - start_time)
        rospy.loginfo(f'[TIMIMG]: {time.time() - start_time}')
        
        n = 0
        for i in indices:
            mask = results[0].masks[i]
            
            segment = []

            for point in mask.xy[0]:
                segment.append(Point32(int(point[0]), int(point[1]), 0.0))
                
            detected_objects[n].segment = segment
            n += 1
            
            mask_raw = mask.cpu().data.numpy().transpose(1, 2, 0)
            
            # Convert single channel grayscale to 3 channel image
            mask_3channel = cv2.cvtColor(mask_raw, cv2.COLOR_GRAY2BGR)

            # Resize the mask to the same size as the image (can probably be removed if image is the same size as the model)
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
        detected_objects_msg.header.stamp = rospy.Time.now()
        depth_msg.header.stamp = rospy.Time.now()
        self.rgb_image.header.stamp = rospy.Time.now()
        odom_msg.header.stamp = rospy.Time.now()
        
        self.rgb_image.header.frame_id = 'map'
        depth_msg.header.frame_id = 'map'
        odom_msg.header.frame_id = 'map'
        
        print('depth: ', depth_msg.header.stamp.nsecs)
        print('image: ', self.rgb_image.header.stamp.nsecs)
        print('odom: ', odom_msg.header.stamp.nsecs)
        print('detected objects: ', detected_objects_msg.header.stamp.nsecs)
        
        # publish depth image and odometry together with segmented image for synchronization
        self.image_pub.publish(self.rgb_image)
        self.depth_pub.publish(depth_msg)
        self.odom_pub.publish(odom_msg)
        self.detected_objects_pub.publish(detected_objects_msg)
        


    def run(self):
        # Keep the node running
        rospy.spin()

if __name__ == '__main__':
    try:
        node = YOLOv9SegNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
