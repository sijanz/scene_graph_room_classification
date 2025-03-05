#!/usr/bin/env python

import sys
import pickle
import networkx
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from scene_graph.msg import GraphObject, GraphObjects, ClassifiedRoom, RoomPolygonList, RoomWithObjects
from std_msgs.msg import String, Int32, Bool
import numpy as np
import math
from geometry_msgs.msg import Point32

# FIXME: ROS has to be installed

class BuildingNode:
    def __init__(self, id, center_point) -> None:
        self.id = id
        self.center_point = center_point
        
class RoomNode:
    def __init__(self, id, class_id, polygon, center_point) -> None:
        self.id = id
        self.class_id = class_id
        self.polygon = polygon
        self.center_point = center_point

class ObjectNode:
    def __init__(self, id, class_id, bounding_box) -> None:
        self.id = id
        self.class_id = class_id
        self.bounding_box = bounding_box
        
class SceneGraphVisualier:
    
    def __init__(self) -> None:
    
        self.objects_pub = rospy.Publisher('/graph_viz/graph_objects', GraphObjects, queue_size=10)
        self.object_bbox_markers_pub = rospy.Publisher('/graph_viz/object_bbox_marker', MarkerArray, queue_size=10)
        self.points_marker_pub = rospy.Publisher('/graph_viz/bbox_points', MarkerArray, queue_size=10)
        self.room_markers_pub = rospy.Publisher('/graph_viz/room_markers', MarkerArray, queue_size=10)
        self.building_markers_pub = rospy.Publisher('/graph_viz/building_markers', MarkerArray, queue_size=10)
        self.line_markers_pub = rospy.Publisher('/graph_viz/line_markers', MarkerArray, queue_size=10)
        self.text_markers_pub = rospy.Publisher('/graph_viz/text_markers', MarkerArray, queue_size=10)

        path = '/home/ros/graph_ws/src/scene_graph/evaluation/scaling/scene_graph_1.pkl'
        self.scene_graph = pickle.load(open(path, 'rb'))
        
        
    def visualize_graph(self):
    
        while not rospy.is_shutdown():
            

                    # publish bounding box markers for each object
            marker_array = MarkerArray()
                    
            bounding_boxes = []
            nodes = list(self.scene_graph.nodes)
            
            for node in nodes:
                
                if type(self.scene_graph.nodes[node]['data']) is ObjectNode:
                    
                    bounding_boxes.append(
                            [
                                self.tuple_from_point(self.scene_graph.nodes[node]['data'].bounding_box[0]),
                                self.tuple_from_point(self.scene_graph.nodes[node]['data'].bounding_box[1]),
                            ]
                    )
                    
                    
            point_marker_array = MarkerArray()
            
            for i, (point1, point2) in enumerate(bounding_boxes):
                marker = self.create_marker_from_bbox(point1, point2, i)
                marker_array.markers.append(marker)
            
            # Publish the marker
            self.object_bbox_markers_pub.publish(marker_array)
            
            bounding_box_points = []
            for bbox in bounding_boxes:
                bounding_box_points.append(bbox[0])
                bounding_box_points.append(bbox[1])
                
            for i, point in enumerate(bounding_box_points):
                point_marker_array.markers.append(self.create_point_marker_from_bbox(point, i))
                
            self.points_marker_pub.publish(point_marker_array)
            
            graph_objects_msg = GraphObjects()
            graph_objects_msg.header.stamp = rospy.Time.now()
            
            graph_objects = []
            for node in nodes:
                if type(self.scene_graph.nodes[node]['data']) is ObjectNode:
                    graph_objects.append(GraphObject(String(self.scene_graph.nodes[node]['data'].class_id), self.scene_graph.nodes[node]['data'].bounding_box))
                    
            graph_objects_msg.objects = graph_objects
            
            self.objects_pub.publish(graph_objects_msg)
            
            # publish building markers
            building_maker_array = MarkerArray()
            
            for i, node in enumerate(nodes):
                if type(self.scene_graph.nodes[node]['data']) is BuildingNode:
                    building_maker_array.markers.append(self.create_building_marker(self.scene_graph.nodes[node]['data'].center_point, i))
            
            self.building_markers_pub.publish(building_maker_array)
            
            # publish room markers
            room_marker_array = MarkerArray()
            
            for i, node in enumerate(nodes):
                if type(self.scene_graph.nodes[node]['data']) is RoomNode:
                    room_marker_array.markers.append(self.create_room_marker(self.scene_graph.nodes[node]['data'].polygon, i))
                    
            self.room_markers_pub.publish(room_marker_array)
            
            # publish line markers
            line_marker_array = MarkerArray()
            
            object_ids = []
            
            # first building level to room level
            i = 0
            for node in nodes:
                if type(self.scene_graph.nodes[node]['data']) is RoomNode:
                    
                    # TODO building center and room center have to be 3d!
                    building_center_point = (self.scene_graph.nodes[0]['data'].center_point[0], self.scene_graph.nodes[0]['data'].center_point[1], 20.0)
                    room_center_point = (self.scene_graph.nodes[node]['data'].center_point[0], self.scene_graph.nodes[node]['data'].center_point[1], 10.0)
                    
                    line_marker_array.markers.append(self.create_line_marker(building_center_point, room_center_point, i))
                    i += 1
                    
                    target_objects = []
                    
                    # get object nodes
                    for edge in list(self.scene_graph.edges):
                        
                        if self.scene_graph.nodes[node]['data'].id == edge[0]: 
                            target_objects.append(edge[1])
                            
                        elif self.scene_graph.nodes[node]['data'].id == edge[1]:
                            target_objects.append(edge[0])
                            
                            
                    for target in target_objects:
                        
                        if type(self.scene_graph.nodes[target]['data']) == BuildingNode:
                            continue
                        
                        # TODO
                        target_center_point = self.calculate_bounding_box_center(self.scene_graph.nodes[target]['data'].bounding_box)#
                        target_center_point = (target_center_point.x, target_center_point.y, target_center_point.z)
                        
                        line_marker_array.markers.append(self.create_line_marker(self.scene_graph.nodes[node]['data'].center_point, target_center_point, i))
                        i += 1
                                            
            self.line_markers_pub.publish(line_marker_array)
            
            # publish text markers
            text_marker_array = MarkerArray()
            
            id = 0
            for node in nodes:
                if type(self.scene_graph.nodes[node]['data']) is BuildingNode:
                    position = (self.scene_graph.nodes[node]['data'].center_point[0], self.scene_graph.nodes[node]['data'].center_point[1], 20.0)
                    text_marker_array.markers.append(self.create_text_marker(position, "building", id))
                    id += 1

                elif type(self.scene_graph.nodes[node]['data']) is RoomNode:
                    position = (self.scene_graph.nodes[node]['data'].center_point[0], self.scene_graph.nodes[node]['data'].center_point[1], 10.0)
                    text_marker_array.markers.append(self.create_text_marker(position, self.scene_graph.nodes[node]['data'].class_id, id))
                    id += 1
                    
                elif type(self.scene_graph.nodes[node]['data']) is ObjectNode:
                    position = self.calculate_bounding_box_center(self.scene_graph.nodes[node]['data'].bounding_box)
                    position = (position.x, position.y, position.z)
                    text_marker_array.markers.append(self.create_text_marker(position, self.scene_graph.nodes[node]['data'].class_id, id))
                    id += 1
                
            self.text_markers_pub.publish(text_marker_array)
            
    def tuple_from_point(self, point):
        return (point.x, point.y, point.z)
    
    def create_marker_from_bbox(self, point1, point2, id):
        marker = Marker()
    
        # Calculate center
        center_x = (point1[0] + point2[0]) / 2.0
        center_y = (point1[1] + point2[1]) / 2.0
        center_z = (point1[2] + point2[2]) / 2.0

        # Calculate dimensions
        size_x = abs(point1[0] - point2[0])
        size_y = abs(point1[1] - point2[1])
        size_z = abs(point1[2] - point2[2])

        # Assume we have two vectors along the edges of the box
        v1 = np.array([point2[0] - point1[0], 0, 0])
        v2 = np.array([0, point2[1] - point1[1], 0])
        v3 = np.array([0, 0, point2[2] - point1[2]])

        # Create orthonormal basis
        x_axis = v1 / np.linalg.norm(v1)
        y_axis = v2 / np.linalg.norm(v2)
        z_axis = v3 / np.linalg.norm(v3)

        # Construct rotation matrix (3x3)
        rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

        # Create homogeneous transformation matrix (4x4)
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix

        # Convert rotation matrix to quaternion
        quaternion = self.quaternion_from_matrix(transformation_matrix)
        
        # Fill in Marker data
        marker.header.frame_id = "map"  # Change frame_id as needed
        marker.header.stamp = rospy.Time.now()
        marker.ns = "bounding_boxes"
        marker.id = id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = center_x
        marker.pose.position.y = center_y
        marker.pose.position.z = center_z
        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]

        marker.scale.x = size_x
        marker.scale.y = size_y
        marker.scale.z = size_z

        marker.color.a = 0.5  # Set transparency (0 = invisible, 1 = opaque)
        marker.color.r = 1.0  # Red color (Change as needed)
        marker.color.g = 0.0
        marker.color.b = 0.0

        return marker   
    
    
    def quaternion_from_matrix(self, matrix):
        """Return quaternion from rotation matrix.

        >>> R = rotation_matrix(0.123, (1, 2, 3))
        >>> q = quaternion_from_matrix(R)
        >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
        True

        """
        q = np.empty((4, ), dtype=np.float64)
        M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
        t = np.trace(M)
        if t > M[3, 3]:
            q[3] = t
            q[2] = M[1, 0] - M[0, 1]
            q[1] = M[0, 2] - M[2, 0]
            q[0] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
        return q
    
    
    def create_point_marker_from_bbox(self, point, id):
        marker = Marker()
    
        # Fill in Marker data
        marker.header.frame_id = "map"  # Change frame_id as needed
        marker.header.stamp = rospy.Time.now()
        marker.ns = "bounding_box_points"
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = point[2]

        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        
        marker.color.a = 1.0  # Set transparency (0 = invisible, 1 = opaque)
        marker.color.r = 0.0  # Red color (Change as needed)
        marker.color.g = 1.0
        marker.color.b = 0.0

        return marker
    
    def create_building_marker(self, center_point, id):
        marker = Marker()
        
        # Fill in Marker data
        marker.header.frame_id = "map"  # Change frame_id as needed
        marker.header.stamp = rospy.Time.now()
        marker.ns = "bounding_box_points"
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = center_point[0]
        marker.pose.position.y = center_point[1]
        marker.pose.position.z = 20.0
        
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        marker.color.a = 1.0  # Set transparency (0 = invisible, 1 = opaque)
        marker.color.r = 1.0  # Red color (Change as needed)
        marker.color.g = 1.0
        marker.color.b = 0.0

        return marker
    
    
    def create_text_marker(self, position, text, id):
        text_marker = Marker()
        text_marker.header.frame_id = "map"  # or any frame you're using
        text_marker.header.stamp = rospy.Time.now()
        text_marker.ns = "labels"
        text_marker.id = id  # Ensure unique IDs for all markers
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position = Point32(
            position[0],
            position[1],
            position[2] + 0.5  # Slightly above 
        )
        text_marker.scale.z = 0.5  # Text size

        # Properly initialize the orientation quaternion for the text marker
        text_marker.pose.orientation.x = 0.0
        text_marker.pose.orientation.y = 0.0
        text_marker.pose.orientation.z = 0.0
        text_marker.pose.orientation.w = 1.0

        # Define the color of the text
        text_marker.color.r = 1.0
        text_marker.color.g = 0.65
        text_marker.color.b = 0.0
        text_marker.color.a = 1.0  # Alpha (transparency)

        # Set the text content
        text_marker.text = text
        
        return text_marker
    
    
    def create_room_marker(self, polygon, id):
        marker = Marker()
        
        # Fill in Marker data
        marker.header.frame_id = "map"  # Change frame_id as needed
        marker.header.stamp = rospy.Time.now()
        marker.ns = "bounding_box_points"
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        center_x = 0.0
        center_y = 0.0
        
        if polygon is not None:
            for point in polygon:
                center_x += point.x
                center_y += point.y
                
            center_x /= len(polygon)
            center_y /= len(polygon)

        marker.pose.position.x = center_x
        marker.pose.position.y = center_y
        marker.pose.position.z = 10.0
        
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        marker.color.a = 1.0  # Set transparency (0 = invisible, 1 = opaque)
        marker.color.r = 0.0  # Red color (Change as needed)
        marker.color.g = 1.0
        marker.color.b = 0.0

        return marker
    
    
    def create_line_marker(self, point1, point2, i):
        
        z1 = 10.0
        z2 = 10.0
        
        if len(point1) > 2:
            z1 = point1[2]
        
        if len(point2) > 2:
            z2 = point2[2]
        
        point1 = Point32(point1[0], point1[1], z1)
        point2 = Point32(point2[0], point2[1], z2)
        
        line_marker = Marker()
        line_marker.header.frame_id = "map"
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = "lines"
        line_marker.id = i
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.04

        line_marker.color.r = 0.0
        line_marker.color.g = 0.0
        line_marker.color.b = 1.0
        line_marker.color.a = 1.0
        
        line_marker.pose.orientation.x = 0.0
        line_marker.pose.orientation.y = 0.0
        line_marker.pose.orientation.z = 0.0
        line_marker.pose.orientation.w = 1.0

        line_marker.points.append(point1)
        line_marker.points.append(point2)
        
        return line_marker
    
    
    def calculate_bounding_box_center(self, bounding_box):
        
        # Initialize sums for x, y, and z coordinates
        sum_x = sum_y = sum_z = 0.0
        
        # Sum all the coordinates
        for point in bounding_box:
            sum_x += point.x
            sum_y += point.y
            sum_z += point.z

        # Calculate the averages
        center_x = sum_x / 2
        center_y = sum_y / 2
        center_z = sum_z / 2

        return Point32(center_x, center_y, center_z)
    

if __name__ == '__main__':
    rospy.init_node('graph_management_node')
    
    try:
        viz = SceneGraphVisualier()
        viz.visualize_graph()
    except rospy.ROSInterruptException:
        pass
