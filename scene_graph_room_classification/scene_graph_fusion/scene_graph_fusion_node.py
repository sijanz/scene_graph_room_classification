#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

import networkx as nx
import numpy as np
import math
from scene_graph_interfaces.msg import Object3DBoundingBox, Object3DBoundingBoxList, ClassifiedRoom, RoomPolygonList
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point32, Point
from scipy.spatial.transform import Rotation as R
import pickle
import time
from shapely.geometry import Polygon
import json


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


class GraphManagementNode(Node):
    def __init__(self) -> None:
        super().__init__('graph_management_node')
        
        # Create subscribers
        self.create_subscription(Object3DBoundingBoxList, '/scene_graph/bounding_boxes_3d', 
                                self.bounding_boxes_callback, 10)
        
        # FIXME: add later
        # self.create_subscription(RoomPolygonList, '/scene_graph/rooms', 
        #                         self.rooms_callback, 10)
        # self.create_subscription(ClassifiedRoom, '/scene_graph/classified_room', 
        #                         self.classified_room_callback, 10)
        # self.create_subscription(Bool, '/scene_graph/control', 
        #                         self.control_callback, 10)
        
        # FIXME: add room functionality later
        # Create publishers
        # self.objects_pub = self.create_publisher(GraphObjects, 'scene_graph/graph_objects', 10)
        # self.room_with_objects_pub = self.create_publisher(RoomWithObjects, 
        #                                                    '/scene/graph/room_with_objects', 10)
        
        # Marker publishers
        self.object_bbox_markers_pub = self.create_publisher(MarkerArray, 
                                                             '/scene_graph/viz/object_bbox_marker', 10)
        self.points_marker_pub = self.create_publisher(MarkerArray, 
                                                       '/scene_graph/debug/bbox_points', 10)
        self.room_markers_pub = self.create_publisher(MarkerArray, 
                                                      '/scene_graph/viz/room_markers', 10)
        self.building_markers_pub = self.create_publisher(MarkerArray, 
                                                          '/scene_graph/viz/building_markers', 10)
        self.line_markers_pub = self.create_publisher(MarkerArray, 
                                                      '/scene_graph/viz/line_markers', 10)
        self.text_markers_pub = self.create_publisher(MarkerArray, 
                                                      '/scene_graph/viz/text_markers', 10)
        
        # Initialize scene graph
        self.scene_graph = nx.Graph()
        self.scene_graph.add_node(0, data=BuildingNode(0, (0.0, 0.0)))
        self.scene_graph.add_node(1, data=RoomNode(1, 'default_room', None, (0.0, 0.0)))
        self.scene_graph.add_edge(0, 1)
        
        self.current_iteration = 1
        self.n = 2
        self.graph_lock = False
        self.rooms_segmented = False
        
        # id, size
        self.rooms = [[1, 999]]
        
        # room where the robot currently is in
        self.current_room_id = 1
        self.rooms_classified = False
        self.overlapping_threshold = 0.8
        self.old_marker_ids = []
        
        # Create timer for main loop (runs at 2 Hz)
        self.timer = self.create_timer(0.5, self.main_loop_callback)
        
        self.get_logger().info('Graph Management Node initialized')
    
    def main_loop_callback(self):
        """Timer callback for main processing loop"""
        
        self.graph_lock = True
        self.publish_markers()
        self.graph_lock = False
        
        # FIXME: add room functionality later
        # nodes = list(self.scene_graph.nodes)
        # for node in nodes:
        #     if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
        #         if type(self.scene_graph.nodes[node]['data']) is RoomNode:
        #             object_names_in_room = []
        #             children = list(self.scene_graph.neighbors(node))
                    
        #             for c in children:
        #                 if c in self.scene_graph and 'data' in self.scene_graph.nodes[c]:
        #                     if type(self.scene_graph.nodes[c]['data']) is ObjectNode:
        #                         object_names_in_room.append(
        #                             String(data=self.scene_graph.nodes[c]['data'].class_id))
                    
                    
        #             room_with_objects_msg = RoomWithObjects()
        #             room_with_objects_msg.header.stamp = self.get_clock().now().to_msg()
        #             room_with_objects_msg.id = Int32(data=node)
        #             room_with_objects_msg.objects = object_names_in_room
        #             self.room_with_objects_pub.publish(room_with_objects_msg)
        
        while self.graph_lock:
            time.sleep(0.01)
        
        self.graph_lock = True
        nodes = list(self.scene_graph.nodes)
        
        # FIXME: why rebuild all edges?
        self.scene_graph.remove_edges_from(list(self.scene_graph.edges))
        
        for object_node in nodes:
            if object_node in self.scene_graph and 'data' in self.scene_graph.nodes[object_node]:
                if type(self.scene_graph.nodes[object_node]['data']) is ObjectNode:
                    min_dist = np.inf
                    min_node = 0
                    
                    for room_node in nodes:
                        if room_node in self.scene_graph and 'data' in self.scene_graph.nodes[room_node]:
                            if type(self.scene_graph.nodes[room_node]['data']) is RoomNode:
                                if (self.scene_graph.nodes[room_node]['data'].polygon is None):
                                    continue
                                
                                if (self.is_object_in_room(
                                    self.get_object_2d_position(self.scene_graph.nodes[object_node]['data']), 
                                    self.scene_graph.nodes[room_node]['data'].polygon)):
                                    object_id = self.scene_graph.nodes[object_node]['data'].id
                                    self.scene_graph.add_edge(room_node, object_id)
                                    break
                
                elif type(self.scene_graph.nodes[object_node]['data']) is RoomNode:
                    for room_node in nodes:
                        if type(self.scene_graph.nodes[room_node]['data']) is RoomNode and room_node != object_node:
                            if self.has_adjacent_points(
                                self.scene_graph.nodes[object_node]['data'].polygon, 
                                self.scene_graph.nodes[room_node]['data'].polygon):
                                self.scene_graph.add_edge(room_node, object_node)
                                
        # for object_node in list(self.scene_graph.nodes):
        #     if object_node in self.scene_graph and 'data' in self.scene_graph.nodes[object_node]:
        #         if type(self.scene_graph.nodes[object_node]['data']) is ObjectNode:
        #             has_edge = False
                    
        #             for edge in list(self.scene_graph.edges):
        #                 if edge[0] == object_node or edge[1] == object_node:
        #                     has_edge = True
                    
        #             if not has_edge:
        #                 min_dist = np.inf
        #                 min_node = 0
                        
        #                 for room_node in list(self.scene_graph.nodes):
        #                     if room_node in self.scene_graph and 'data' in self.scene_graph.nodes[room_node]:
        #                         if type(self.scene_graph.nodes[room_node]['data']) is RoomNode:
        #                             dist = self.euclidean_distance2d(
        #                                 self.get_object_2d_position(self.scene_graph.nodes[object_node]['data']), 
        #                                 self.scene_graph.nodes[room_node]['data'].center_point)
        #                             if dist < min_dist:
        #                                 min_dist = dist
        #                                 min_node = room_node
                        
        #                 self.scene_graph.add_edge(min_node, object_node)
        
        self.publish_markers()
        self.graph_lock = False
    
    def control_callback(self, msg):
        """Save and delete graph"""
        if msg.data == True:
            time.sleep(5.0)
            pickle.dump(self.scene_graph, open(f"scene_graph_{self.current_iteration}.pkl", 'wb'))
            self.current_iteration += 1
            
            self.scene_graph = nx.Graph()
            self.scene_graph.add_node(0, data=BuildingNode(0, (0.0, 0.0)))
            self.scene_graph.add_node(1, data=RoomNode(1, 'default_room', None, (0.0, 0.0)))
            self.scene_graph.add_edge(0, 1)
            
            self.graph_lock = False
            self.current_room_id = 1
            self.rooms_classified = False
            self.rooms_segmented = False
            self.rooms = [[1, 999]]
    
    def classified_room_callback(self, msg):
        nodes = list(self.scene_graph.nodes)
        for room in self.rooms:
            if self.scene_graph.nodes[room[0]]['data'].id == msg.id.data:
                self.scene_graph.nodes[room[0]]['data'].class_id = msg.label.data
                break
    
    def bounding_boxes_callback(self, msg):
        """Callback for detected objects"""
        
        # to disable interrupting the constructor
        if len(list(self.scene_graph.nodes)) < 2:
            return
        
        while self.graph_lock:
            time.sleep(0.01)
        
        self.graph_lock = True
        changed_room_ids = []
        
        for object in msg.bbox:
            in_graph = self.is_object_in_graph(object)
            
            if in_graph == -1:
                self.scene_graph.add_node(self.n, data=ObjectNode(
                    self.n, object.name.data, object.bounding_box))
                
                if not self.rooms_classified:
                    
                    # add to default room first
                    self.scene_graph.add_edge(self.current_room_id, self.n)
                else:
                    nodes = list(self.scene_graph.nodes)
                    for room in self.rooms:
                        if self.is_object_in_room(
                            self.get_object_2d_position(object), 
                            self.scene_graph.nodes[room[0]]['data'].polygon):
                            self.scene_graph.add_edge(
                                self.scene_graph.nodes[room[0]]['data'].id, self.n)
                            if not self.scene_graph.nodes[room[0]]['data'].id in changed_room_ids:
                                changed_room_ids.append(self.scene_graph.nodes[room[0]]['data'].id)
                            break
                
                self.n += 1
            else:
                merged_box = self.merge_bounding_boxes(
                    self.scene_graph.nodes[in_graph]['data'].bounding_box[0], 
                    self.scene_graph.nodes[in_graph]['data'].bounding_box[1],
                    object.bounding_box[0], object.bounding_box[1])
                self.scene_graph.nodes[in_graph]['data'].bounding_box[0] = merged_box[0]
                self.scene_graph.nodes[in_graph]['data'].bounding_box[1] = merged_box[1]
        
        self.graph_lock = False
    
    def publish_markers(self):
        """Publishing of markers"""
        
        # Publish bounding box markers for each object
        marker_array = MarkerArray()
        bounding_boxes = []
        nodes = list(self.scene_graph.nodes)
        
        for node in nodes:
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is ObjectNode:
                    bounding_boxes.append((
                        self.tuple_from_point(self.scene_graph.nodes[node]['data'].bounding_box[0]),
                        self.tuple_from_point(self.scene_graph.nodes[node]['data'].bounding_box[1])
                    ))
        
        for i, (point1, point2) in enumerate(bounding_boxes):
            marker = self.create_marker_from_bbox(point1, point2, i)
            marker_array.markers.append(marker)
            
        # FIXME: why publish object markers?        
        # graph_objects_msg = GraphObjects()
        # graph_objects_msg.header.stamp = self.get_clock().now().to_msg()
        # graph_objects = []
        
        # for node in nodes:
        #     if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
        #         if type(self.scene_graph.nodes[node]['data']) is ObjectNode:
        #             graph_objects.append(GraphObject(
        #                 String(data=self.scene_graph.nodes[node]['data'].class_id), 
        #                 self.scene_graph.nodes[node]['data'].bounding_box))
        
        # graph_objects_msg.objects = graph_objects
        # self.objects_pub.publish(graph_objects_msg)
        
        # Publish building markers
        building_maker_array = MarkerArray()
        for i, node in enumerate(nodes):
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is BuildingNode:
                    building_maker_array.markers.append(
                        self.create_building_marker(
                            self.scene_graph.nodes[node]['data'].center_point, i))
        
        delete_marker_array = MarkerArray()
        for i in self.old_marker_ids:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "room_markers"
            marker.id = i
            marker.action = Marker.DELETE
            marker_array.markers.append(marker)
        
        self.room_markers_pub.publish(delete_marker_array)
        self.old_marker_ids = []
        
        # Publish room markers
        room_marker_array = MarkerArray()
        for i, node in enumerate(nodes):
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is RoomNode:
                    room_marker_array.markers.append(
                        self.create_room_marker(
                            self.scene_graph.nodes[node]['data'].polygon, i))
                    self.old_marker_ids.append(i)
                    
                    
        # Publish line markers
        line_marker_array = MarkerArray()
        
        # First building level to room level
        i = 0
        for node in nodes:
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is RoomNode:
                    building_center_point = (
                        self.scene_graph.nodes[0]['data'].center_point[0], 
                        self.scene_graph.nodes[0]['data'].center_point[1], 15.0)
                    room_center_point = (
                        self.scene_graph.nodes[node]['data'].center_point[0], 
                        self.scene_graph.nodes[node]['data'].center_point[1], 8.0)
                    line_marker_array.markers.append(
                        self.create_line_marker(building_center_point, room_center_point, i))
                    i += 1
                    
                    target_objects = []
                    
                    # get object nodes
                    for edge in list(self.scene_graph.edges):
                        if self.scene_graph.nodes[node]['data'].id == edge[0]:
                            target_objects.append(edge[1])
                        elif self.scene_graph.nodes[node]['data'].id == edge[1]:
                            target_objects.append(edge[0])
                    
                    for target in target_objects:
                        if target in self.scene_graph and 'data' in self.scene_graph.nodes[target]:
                            if type(self.scene_graph.nodes[target]['data']) == BuildingNode or \
                               type(self.scene_graph.nodes[target]['data']) == RoomNode:
                                continue
                            
                            target_center_point = self.calculate_bounding_box_center(
                                self.scene_graph.nodes[target]['data'].bounding_box)
                            target_center_point = (
                                target_center_point.x, target_center_point.y, target_center_point.z)
                            line_marker_array.markers.append(
                                self.create_line_marker(
                                    self.scene_graph.nodes[node]['data'].center_point, 
                                    target_center_point, i))
                            i += 1
        
        for edge in list(self.scene_graph.edges):
            if type(self.scene_graph.nodes[edge[0]]['data']) == RoomNode and \
               type(self.scene_graph.nodes[edge[1]]['data']) == RoomNode:
                room_center_point_1 = (
                    self.scene_graph.nodes[edge[0]]['data'].center_point[0], 
                    self.scene_graph.nodes[edge[0]]['data'].center_point[1], 8.0)
                room_center_point_2 = (
                    self.scene_graph.nodes[edge[1]]['data'].center_point[0], 
                    self.scene_graph.nodes[edge[1]]['data'].center_point[1], 8.0)
                line_marker_array.markers.append(
                    self.create_line_marker(room_center_point_1, room_center_point_2, i))
                i += 1
                
        # Publish text markers
        text_marker_array = MarkerArray()
        id = 0
        
        for node in nodes:
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is BuildingNode:
                    position = (
                        self.scene_graph.nodes[node]['data'].center_point[0], 
                        self.scene_graph.nodes[node]['data'].center_point[1], 15.0)
                    text_marker_array.markers.append(
                        self.create_text_marker(position, "building", id))
                    id += 1
                elif type(self.scene_graph.nodes[node]['data']) is RoomNode:
                    position = (
                        self.scene_graph.nodes[node]['data'].center_point[0], 
                        self.scene_graph.nodes[node]['data'].center_point[1], 8.0)
                    text_marker_array.markers.append(
                        self.create_text_marker(
                            position, self.scene_graph.nodes[node]['data'].class_id, id))
                    id += 1
                elif type(self.scene_graph.nodes[node]['data']) is ObjectNode:
                    position = self.calculate_bounding_box_center(
                        self.scene_graph.nodes[node]['data'].bounding_box)
                    position = (position.x, position.y, position.z)
                    text_marker_array.markers.append(
                        self.create_text_marker(
                            position, self.scene_graph.nodes[node]['data'].class_id, id))
                    id += 1
                    
        # Publish all markers
        self.object_bbox_markers_pub.publish(marker_array)
        # self.objects_pub.publish(graph_objects_msg)
        self.building_markers_pub.publish(building_maker_array)
        self.room_markers_pub.publish(room_marker_array)
        self.line_markers_pub.publish(line_marker_array)
        self.text_markers_pub.publish(text_marker_array)
    
    def merge_bounding_boxes(self, min1, max1, min2, max2):
        """Merges two axis-aligned 3D bounding boxes"""
        
        # FIXME: merging doesn't work as intended
        min1 = (min1.x, min1.y, min1.z)
        min2 = (min2.x, min2.y, min2.z)
        max1 = (max1.x, max1.y, max1.z)
        max2 = (max2.x, max2.y, max2.z)
        
        merged_min = (
            min(min1[0], min2[0]),
            min(min1[1], min2[1]),
            min(min1[2], min2[2])
        )
        
        merged_max = (
            max(max1[0], max2[0]),
            max(max1[1], max2[1]),
            max(max1[2], max2[2])
        )
        
        merged_min = Point32(x=merged_min[0], y=merged_min[1], z=merged_min[2])
        merged_max = Point32(x=merged_max[0], y=merged_max[1], z=merged_max[2])
        
        return merged_min, merged_max
    
    def create_point_marker_from_bbox(self, point, id):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
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
        
        marker.color.a = 0.7
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        
        return marker
    
    def create_building_marker(self, center_point, id):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "bounding_box_points"
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.lifetime = Duration(seconds=0, nanoseconds=600000000).to_msg()
        
        marker.pose.position.x = center_point[0]
        marker.pose.position.y = center_point[1]
        marker.pose.position.z = 15.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        marker.color.a = 1.0
        marker.color.r = 0.25
        marker.color.g = 0.60
        marker.color.b = 0.84
        
        return marker
    
    def create_text_marker(self, position, text, id):
        text_marker = Marker()
        text_marker.header.frame_id = "map"
        text_marker.header.stamp = self.get_clock().now().to_msg()
        text_marker.ns = "labels"
        text_marker.id = id
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.lifetime = Duration(seconds=0, nanoseconds=600000000).to_msg()
        
        pos = Point()
        pos.x = position[0]
        pos.y = position[1]
        pos.z = position[2] + 0.8
        
        text_marker.pose.position = pos
        
        text_marker.scale.z = 0.7
        
        text_marker.pose.orientation.x = 0.0
        text_marker.pose.orientation.y = 0.0
        text_marker.pose.orientation.z = 0.0
        text_marker.pose.orientation.w = 1.0
        
        text_marker.color.r = 0.25
        text_marker.color.g = 0.60
        text_marker.color.b = 0.84
        text_marker.color.a = 1.0
        
        text_marker.text = text
        
        return text_marker
    
    def create_room_marker(self, polygon, id):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "room_markers"
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.lifetime = Duration(seconds=0, nanoseconds=600000000).to_msg()
        
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
        marker.pose.position.z = 8.0
        
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        
        marker.color.a = 1.0
        marker.color.r = 0.25
        marker.color.g = 0.60
        marker.color.b = 0.84
        
        return marker
    
    def create_line_marker(self, point1, point2, i):
        z1 = 8.0
        z2 = 8.0
        
        if len(point1) > 2:
            z1 = point1[2]
        if len(point2) > 2:
            z2 = point2[2]
            
        point1 = Point(x=point1[0], y=point1[1], z=z1)
        point2 = Point(x=point2[0], y=point2[1], z=z2)
        
        line_marker = Marker()
        line_marker.header.frame_id = "map"
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = "lines"
        line_marker.id = i
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.04
        line_marker.lifetime = Duration(seconds=0, nanoseconds=600000000).to_msg()
        
        line_marker.color.r = 0.25
        line_marker.color.g = 0.60
        line_marker.color.b = 0.84
        line_marker.color.a = 0.7
        
        line_marker.pose.orientation.x = 0.0
        line_marker.pose.orientation.y = 0.0
        line_marker.pose.orientation.z = 0.0
        line_marker.pose.orientation.w = 1.0
        
        line_marker.points.append(point1)
        line_marker.points.append(point2)
        
        return line_marker
    
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
        
        # Create orthonormal basis
        v1 = np.array([point2[0] - point1[0], 0, 0])
        v2 = np.array([0, point2[1] - point1[1], 0])
        v3 = np.array([0, 0, point2[2] - point1[2]])
        
        x_axis = v1 / np.linalg.norm(v1)
        y_axis = v2 / np.linalg.norm(v2)
        z_axis = v3 / np.linalg.norm(v3)
        
        # Construct rotation matrix
        rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
        
        # Create homogeneous transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        
        # Convert rotation matrix to quaternion using scipy
        rotation_obj = R.from_matrix(rotation_matrix)
        quaternion = rotation_obj.as_quat()  # Returns [x, y, z, w]
        
        # Fill in Marker data
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "bounding_boxes"
        marker.id = id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.lifetime = Duration(seconds=0, nanoseconds=600000000).to_msg()
        
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
        
        marker.color.a = 0.5
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        
        return marker
    
    def rooms_callback(self, msg):
        """Process room segmentation"""
        while self.graph_lock:
            time.sleep(0.01)
        
        self.graph_lock = True
        
        for node in list(self.scene_graph.nodes):
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is RoomNode:
                    self.scene_graph.remove_node(node)
        
        self.rooms = []
        self.rooms_classified = True
        self.scene_graph.remove_edges_from(list(self.scene_graph.edges))
        
        for room in msg.rooms:
            center_x = 0.0
            center_y = 0.0
            
            if room.points is not None:
                for point in room.points:
                    center_x += point.x
                    center_y += point.y
                center_x /= len(room.points)
                center_y /= len(room.points)
            
            self.scene_graph.add_node(self.n, data=RoomNode(
                self.n, '-', room.points, (center_x, center_y)))
            self.scene_graph.add_edge(0, self.n)
            self.rooms.append([self.n, self.polygon_area(room.points)])
            self.n += 1
        
        # Sort rooms list
        self.rooms = sorted(self.rooms, key=lambda x: x[1])
        
        # Calculate building center
        center_x = 0.0
        center_y = 0.0
        room_count = 0
        nodes = list(self.scene_graph.nodes)
        
        for node in nodes:
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is RoomNode:
                    center_x += self.scene_graph.nodes[node]['data'].center_point[0]
                    center_y += self.scene_graph.nodes[node]['data'].center_point[1]
                    room_count += 1
        
        center_x /= room_count
        center_y /= room_count
        self.scene_graph.nodes[0]['data'].center_point = (center_x, center_y)
        
        # Redo all edges
        nodes = list(self.scene_graph.nodes)
        for object_node in nodes:
            if object_node in self.scene_graph and 'data' in self.scene_graph.nodes[object_node]:
                if type(self.scene_graph.nodes[object_node]['data']) is ObjectNode:
                    min_dist = np.inf
                    min_node = 0
                    
                    for room_node in nodes:
                        if room_node in self.scene_graph and 'data' in self.scene_graph.nodes[room_node]:
                            if type(self.scene_graph.nodes[room_node]['data']) is RoomNode:
                                dist = self.euclidean_distance2d(
                                    self.get_object_2d_position(self.scene_graph.nodes[object_node]['data']), 
                                    self.scene_graph.nodes[room_node]['data'].center_point)
                                if dist < min_dist:
                                    min_dist = dist
                                    min_node = room_node
                    
                    self.scene_graph.add_edge(min_node, object_node)
        
        self.rooms_segmented = True
        self.graph_lock = False
    
    def is_object_in_room(self, point, vertices):
        """Determine if a point is inside a given polygon"""
        point = Point32(x=point[0], y=point[1], z=0.0)
        winding_number = 0
        n = len(vertices)
        
        for i in range(n):
            V_i = vertices[i]
            V_next = vertices[(i + 1) % n]
            
            if V_i.y <= point.y:
                if V_next.y > point.y:
                    is_left = ((V_next.x - V_i.x) * (point.y - V_i.y) -
                              (point.x - V_i.x) * (V_next.y - V_i.y))
                    if is_left > 0:
                        winding_number += 1
            else:
                if V_next.y <= point.y:
                    is_left = ((V_next.x - V_i.x) * (point.y - V_i.y) -
                              (point.x - V_i.x) * (V_next.y - V_i.y))
                    if is_left < 0:
                        winding_number -= 1
        
        return winding_number != 0
    
    def get_object_2d_position(self, object):
        if hasattr(object, 'bounding_box'):
            center_x = (object.bounding_box[0].x + object.bounding_box[1].x) / 2
            center_y = (object.bounding_box[0].y + object.bounding_box[1].y) / 2
        else:
            center_x = (object[0].x + object[1].x) / 2
            center_y = (object[0].y + object[1].y) / 2
        return (center_x, center_y)
    
    def is_object_in_graph(self, object):
        nodes = list(self.scene_graph.nodes)
        center_point = self.calculate_bounding_box_center(object.bounding_box)
        center_point = [center_point.x, center_point.y, center_point.z]
        
        for node in nodes:
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is ObjectNode:
                    overlap_volume = self.overlap_volume(
                        object.bounding_box, 
                        self.scene_graph.nodes[node]['data'].bounding_box)
                    
                    if (overlap_volume / self.bounding_box_volume(object.bounding_box) > self.overlapping_threshold or 
                        overlap_volume / self.bounding_box_volume(self.scene_graph.nodes[node]['data'].bounding_box) > self.overlapping_threshold):
                        return node
        
        return -1
    
    def is_point_inside_bbox(self, point1, point2, P):
        x1, y1, z1 = point1.x, point1.y, point1.z
        x2, y2, z2 = point2.x, point2.y, point2.z
        px, py, pz = P.x, P.y, P.z
        
        inside_x = min(x1, x2) <= px <= max(x1, x2)
        inside_y = min(y1, y2) <= py <= max(y1, y2)
        inside_z = min(z1, z2) <= pz <= max(z1, z2)
        
        return inside_x and inside_y and inside_z
    
    def calculate_bounding_box_center(self, bounding_box):
        sum_x = sum_y = sum_z = 0.0
        
        for point in bounding_box:
            sum_x += point.x
            sum_y += point.y
            sum_z += point.z
        
        center_x = sum_x / 2
        center_y = sum_y / 2
        center_z = sum_z / 2
        
        return Point32(x=center_x, y=center_y, z=center_z)
    
    def euclidean_distance(self, point_1, point_2):
        return math.sqrt((point_2.x - point_1.x)**2 + 
                        (point_2.y - point_1.y)**2 + 
                        (point_2.z - point_1.z)**2)
    
    def euclidean_distance2d(self, point_1, point_2):
        return math.sqrt((point_2[0] - point_1[0])**2 + 
                        (point_2[1] - point_1[1])**2)
    
    def tuple_from_point(self, point):
        return (point.x, point.y, point.z)
    
    def calculate_center(self, corners):
        """Calculate the center of the bounding box given its 8 corner points"""
        corners_np = np.array(corners)
        center = np.mean(corners_np, axis=0)
        return tuple(center)
    
    def calculate_orientation(self, corners):
        """Calculate the orientation of the bounding box"""
        vec_x = np.array(corners[1]) - np.array(corners[0])
        vec_y = np.array(corners[3]) - np.array(corners[0])
        vec_z = np.array(corners[4]) - np.array(corners[0])
        
        vec_x /= np.linalg.norm(vec_x)
        vec_y /= np.linalg.norm(vec_y)
        vec_z /= np.linalg.norm(vec_z)
        
        rotation_matrix = np.identity(4)
        rotation_matrix[0:3, 0] = vec_x
        rotation_matrix[0:3, 1] = vec_y
        rotation_matrix[0:3, 2] = vec_z
        
        rotation_obj = R.from_matrix(rotation_matrix[:3, :3])
        quaternion = rotation_obj.as_quat()
        
        return self.normalize_quaternion(quaternion)
    
    def normalize_quaternion(self, quat):
        """Normalize a quaternion"""
        norm = np.linalg.norm(quat)
        if norm == 0:
            self.get_logger().warn("Zero norm quaternion, cannot normalize!")
            return (0, 0, 0, 1)
        return tuple(np.array(quat) / norm)
    
    def calculate_scale(self, corners):
        """Calculate the scale (dimensions) of the bounding box"""
        scale_x = np.linalg.norm(np.array(corners[1]) - np.array(corners[0]))
        scale_y = np.linalg.norm(np.array(corners[3]) - np.array(corners[0]))
        scale_z = np.linalg.norm(np.array(corners[4]) - np.array(corners[0]))
        return (scale_x, scale_y, scale_z)
    
    def bounding_box_volume(self, bounding_box):
        """Calculate the volume of a 3D bounding box"""
        min_point = [bounding_box[0].x, bounding_box[0].y, bounding_box[0].z]
        max_point = [bounding_box[1].x, bounding_box[1].y, bounding_box[1].z]
        
        length_x = max_point[0] - min_point[0]
        length_y = max_point[1] - min_point[1]
        length_z = max_point[2] - min_point[2]
        
        volume = length_x * length_y * length_z
        return volume
    
    def overlap_volume(self, bounding_box1, bounding_box2):
        """Calculate the volume of the intersection of two 3D bounding boxes"""
        box1_min = [bounding_box1[0].x, bounding_box1[0].y, bounding_box1[0].z]
        box1_max = [bounding_box1[1].x, bounding_box1[1].y, bounding_box1[1].z]
        box2_min = [bounding_box2[0].x, bounding_box2[0].y, bounding_box2[0].z]
        box2_max = [bounding_box2[1].x, bounding_box2[1].y, bounding_box2[1].z]
        
        x_overlap = max(0, min(box1_max[0], box2_max[0]) - max(box1_min[0], box2_min[0]))
        y_overlap = max(0, min(box1_max[1], box2_max[1]) - max(box1_min[1], box2_min[1]))
        z_overlap = max(0, min(box1_max[2], box2_max[2]) - max(box1_min[2], box2_min[2]))
        
        overlap_vol = x_overlap * y_overlap * z_overlap
        return overlap_vol
    
    def polygon_area(self, vertices):
        """Calculate polygon area using Shoelace formula"""
        n = len(vertices)
        area = 0
        
        for i in range(n):
            x1 = vertices[i].x
            y1 = vertices[i].y
            x2 = vertices[(i + 1) % n].x
            y2 = vertices[(i + 1) % n].y
            area += x1 * y2 - y1 * x2
        
        return abs(area) / 2
    
    def has_adjacent_points(self, polygon_1, polygon_2):
        for i in range(0, len(polygon_1), 10):
            for j in range(0, len(polygon_2), 10):
                point_1 = (polygon_1[i].x, polygon_1[i].y, polygon_1[i].z)
                point_2 = (polygon_2[j].x, polygon_2[j].y, polygon_2[j].z)
                if self.euclidean_distance2d(point_1, point_2) < 0.5:
                    return True
        return False
    
    def export_scene_graph_to_json(self, filename):
        """Export the current scene graph to a JSON file"""
        building_node = None
        for node in self.scene_graph.nodes:
            data = self.scene_graph.nodes[node].get('data')
            if isinstance(data, BuildingNode):
                building_node = data
                break
        
        building = {
            "id": f"building_{building_node.id if building_node else '-'}",
            "label": "-",
            "bounding_box": {
                "min_corner": {"x": 0.0, "y": 0.0, "z": 0.0},
                "max_corner": {"x": 0.0, "y": 0.0, "z": 0.0}
            },
            "floors": []
        }
        
        floor = {
            "id": "floor_0",
            "parent_id": building["id"],
            "label": "-",
            "centroide": {"x": building_node.center_point[0], 
                         "y": building_node.center_point[1], "z": 0.0},
            "bounding_box": {
                "min_corner": {"x": 0.0, "y": 0.0, "z": 0.0},
                "max_corner": {"x": 0.0, "y": 0.0, "z": 0.0}
            },
            "rooms": []
        }
        
        for node in self.scene_graph.nodes:
            data = self.scene_graph.nodes[node].get('data')
            if isinstance(data, RoomNode):
                room = {
                    "id": f"room_{data.id}",
                    "parent_id": floor["id"],
                    "label": data.class_id,
                    "centroide": {
                        "x": data.center_point[0],
                        "y": data.center_point[1],
                        "z": 0.0
                    },
                    "bounding_box": {
                        "min_corner": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "max_corner": {"x": 0.0, "y": 0.0, "z": 0.0}
                    },
                    "objects": []
                }
                
                for neighbor in self.scene_graph.neighbors(node):
                    obj_data = self.scene_graph.nodes[neighbor].get('data')
                    if isinstance(obj_data, ObjectNode):
                        obj = {
                            "id": f"obj_{obj_data.id}",
                            "parent_id": room["id"],
                            "semantic_label": obj_data.class_id,
                            "confidence": 1.0,
                            "centroide": {
                                "x": (obj_data.bounding_box[0].x + obj_data.bounding_box[1].x) / 2,
                                "y": (obj_data.bounding_box[0].y + obj_data.bounding_box[1].y) / 2,
                                "z": (obj_data.bounding_box[0].z + obj_data.bounding_box[1].z) / 2
                            },
                            "bounding_box": {
                                "min_corner": {
                                    "x": obj_data.bounding_box[0].x,
                                    "y": obj_data.bounding_box[0].y,
                                    "z": obj_data.bounding_box[0].z
                                },
                                "max_corner": {
                                    "x": obj_data.bounding_box[1].x,
                                    "y": obj_data.bounding_box[1].y,
                                    "z": obj_data.bounding_box[1].z
                                }
                            },
                            "attributes": []
                        }
                        room["objects"].append(obj)
                
                floor["rooms"].append(room)
        
        building["floors"].append(floor)
        
        scene_graph_json = {
            "scene_id": "generated_scene",
            "method_name": "GraphManagementNode_Export",
            "building": building
        }
        
        with open(filename, "w") as f:
            json.dump(scene_graph_json, f, indent=2)
        
        self.get_logger().info(f"Scene graph exported to {filename}")


def main(args=None):
    rclpy.init(args=args)
    node = GraphManagementNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
