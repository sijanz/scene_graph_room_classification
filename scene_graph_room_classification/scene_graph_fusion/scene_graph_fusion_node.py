#!/usr/bin/env python3

# ============================================================================
# SCENE GRAPH FUSION NODE
# ============================================================================
# This node creates and maintains a hierarchical scene graph representing
# the spatial relationship between buildings, rooms, and objects in a 3D
# environment. It uses NetworkX for graph management and performs:
# - Object tracking and merging using bounding box overlap
# - Spatial containment reasoning (objects in rooms)
# - Graph-based visualization in RViz
# - Room segmentation and classification integration
# ============================================================================

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


# ============================================================================
# NODE CLASSES - Scene Graph Hierarchy
# ============================================================================

class BuildingNode:
    """
    Represents a building in the scene graph hierarchy.

    This is the root node of the scene graph, containing all rooms and objects.
    The building node tracks the overall center point of the structure.

    Attributes:
        id (int): Unique identifier for the building node
        center_point (tuple): (x, y) coordinates of building center
    """
    
    def __init__(self, id, center_point) -> None:
        self.id = id
        self.center_point = center_point


class RoomNode:
    """
    Represents a room in the scene graph hierarchy.

    Rooms are intermediate nodes that contain multiple objects and connect
    to the building root. They are defined by polygonal boundaries.

    Attributes:
        id (int): Unique identifier for the room node
        class_id (str): Room classification label (e.g., 'kitchen', 'bedroom')
        polygon (list): List of Point32 defining room boundary vertices
        center_point (tuple): (x, y) coordinates of room centroid
    """
    
    def __init__(self, id, class_id, polygon, center_point) -> None:
        self.id = id
        self.class_id = class_id
        self.polygon = polygon
        self.center_point = center_point


class ObjectNode:
    """
    Represents a detected object in the scene graph hierarchy.

    Objects are leaf nodes in the graph, connected to their containing room.
    Each object is defined by a 3D axis-aligned bounding box.

    Attributes:
        id (int): Unique identifier for the object node
        class_id (str): Object class label (e.g., 'chair', 'table')
        bounding_box (list): [min_corner, max_corner] as Point32 objects
    """
    
    def __init__(self, id, class_id, bounding_box) -> None:
        self.id = id
        self.class_id = class_id
        self.bounding_box = bounding_box


class GraphManagementNode(Node):
    """
    ROS2 node for managing a hierarchical scene graph of a 3D environment.

    This node maintains a NetworkX graph with three levels:
    1. Building (root)
    2. Rooms (intermediate nodes)
    3. Objects (leaf nodes)

    Key functionalities:
    - Receives 3D bounding boxes and adds/updates objects in the graph
    - Performs object association using bounding box overlap (IOU-based)
    - Maintains spatial relationships (containment, adjacency)
    - Publishes visualization markers for RViz
    - Supports room segmentation and classification (partially implemented)
    - Can export scene graph to JSON format
    """
    
    def __init__(self) -> None:
        """
        Initialize the GraphManagementNode.

        Sets up:
        - Subscribers for object detections, room polygons, and control signals
        - Publishers for visualization markers and graph data
        - Initial scene graph with building and default room nodes
        - Timer for periodic marker publishing at 2 Hz
        """
        
        # Initialize the parent Node class
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
        
        
        # ====================================================================
        # PUBLISHERS - Visualization Markers
        # ====================================================================
        
        # Publish object bounding box markers (semi-transparent red cubes)
        self.object_bbox_markers_pub = self.create_publisher(MarkerArray, 
                                                             '/scene_graph/viz/object_bbox_marker', 10)
        
        # Publish debug markers showing bounding box corner points
        self.points_marker_pub = self.create_publisher(MarkerArray, 
                                                       '/scene_graph/debug/bbox_points', 10)
        
        # Publish room markers (spheres at room centers)
        self.room_markers_pub = self.create_publisher(MarkerArray, 
                                                      '/scene_graph/viz/room_markers', 10)
        
        # Publish building marker (sphere at building center), root node
        self.building_markers_pub = self.create_publisher(MarkerArray, 
                                                          '/scene_graph/viz/building_markers', 10)
        
        # Publish line markers showing graph edges (relationships)
        self.line_markers_pub = self.create_publisher(MarkerArray, 
                                                      '/scene_graph/viz/line_markers', 10)
        
        # Publish text labels for all nodes
        self.text_markers_pub = self.create_publisher(MarkerArray, 
                                                      '/scene_graph/viz/text_markers', 10)
        
        # ====================================================================
        # SCENE GRAPH INITIALIZATION
        # ====================================================================
        
        # Create NetworkX undirected graph to store scene structure
        self.scene_graph = nx.Graph()

        # Add root building node (ID=0) at origin
        self.scene_graph.add_node(0, data=BuildingNode(0, (0.0, 0.0)))

        # Add default room node (ID=1) for objects before room segmentation
        self.scene_graph.add_node(1, data=RoomNode(1, 'default_room', None, (0.0, 0.0)))

        # Connect building to default room
        self.scene_graph.add_edge(0, 1)
        
        # ====================================================================
        # STATE VARIABLES
        # ====================================================================

        # Counter for iterations (used for saving multiple scene graphs)
        self.current_iteration = 1

        # Next available node ID (starts at 2, since 0 and 1 are taken)
        self.n = 2

        # Lock to prevent concurrent graph modifications
        self.graph_lock = False

        # Flag indicating if rooms have been segmented
        self.rooms_segmented = False

        # List of [room_id, room_area] pairs for sorting rooms by size
        self.rooms = [[1, 999]]  # Default room with large area

        # ID of the room where the robot is currently located
        self.current_room_id = 1

        # Flag indicating if room types have been classified
        self.rooms_classified = False

        # Intersection-over-Union threshold for object association
        # Objects with >80% overlap are considered the same object
        self.overlapping_threshold = 0.8

        # Track marker IDs for deletion
        self.old_marker_ids = []

        # ====================================================================
        # TIMER FOR PERIODIC UPDATES
        # ====================================================================

        # Run main loop at 2 Hz (every 0.5 seconds)
        # This publishes visualization markers and updates graph structure
        self.timer = self.create_timer(0.5, self.main_loop_callback)

        self.get_logger().info('Graph Management Node initialized')

    
    def main_loop_callback(self):
        """
        Timer callback that runs at 2 Hz for periodic graph maintenance.

        This function performs two main tasks:
        1. Publishes visualization markers for the current graph state
        2. Rebuilds graph edges based on spatial relationships:
           - Objects are connected to their containing room
           - Adjacent rooms are connected to each other
           - All rooms are connected to the building root

        The edge rebuilding is necessary because object positions may have
        been updated (via bounding box merging) since edges were last created.

        Note: The FIXME comment questions whether rebuilding ALL edges every
        0.5 seconds is efficient. A more optimized approach would track which
        nodes have changed and only update relevant edges.
        """
        
        # Acquire lock and publish current visualization
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
        
        # Wait for any ongoing operations to complete
        while self.graph_lock:
            time.sleep(0.01)
        
        # Acquire lock for graph structure modification
        self.graph_lock = True
        
        # Get current node list (may change during iteration)
        nodes = list(self.scene_graph.nodes)
        
        # FIXME: why rebuild all edges?
        self.scene_graph.remove_edges_from(list(self.scene_graph.edges))
        
        # Iterate through all nodes to establish relationships
        for object_node in nodes:
            # Verify node still exists and has data
            if object_node in self.scene_graph and 'data' in self.scene_graph.nodes[object_node]:

                # ============================================================
                # CASE 1: Object Nodes - Find containing room
                # ============================================================
                if type(self.scene_graph.nodes[object_node]['data']) is ObjectNode:
                    min_dist = np.inf
                    min_node = 0

                    # Check each room to find which one contains this object
                    for room_node in nodes:
                        if room_node in self.scene_graph and 'data' in self.scene_graph.nodes[room_node]:
                            if type(self.scene_graph.nodes[room_node]['data']) is RoomNode:
                                # Skip rooms without polygon boundaries
                                if (self.scene_graph.nodes[room_node]['data'].polygon is None):
                                    continue

                                # Use point-in-polygon test (winding number algorithm)
                                if (self.is_object_in_room(
                                    self.get_object_2d_position(self.scene_graph.nodes[object_node]['data']),
                                    self.scene_graph.nodes[room_node]['data'].polygon)):

                                    # Object is inside this room - create edge
                                    object_id = self.scene_graph.nodes[object_node]['data'].id
                                    self.scene_graph.add_edge(room_node, object_id)
                                    break

                # ============================================================
                # CASE 2: Room Nodes - Find adjacent rooms
                # ============================================================
                elif type(self.scene_graph.nodes[object_node]['data']) is RoomNode:
                    # Check all other rooms for adjacency
                    for room_node in nodes:
                        if type(self.scene_graph.nodes[room_node]['data']) is RoomNode and room_node != object_node:
                            # Check if room boundaries share close points (< 0.5m apart)
                            if self.has_adjacent_points(
                                self.scene_graph.nodes[object_node]['data'].polygon,
                                self.scene_graph.nodes[room_node]['data'].polygon):
                                # Rooms are adjacent - create edge
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
        
        # Publish updated visualization
        self.publish_markers()

        # Release lock
        self.graph_lock = False
        
    
    def control_callback(self, msg):
        """
        Callback for control signals to save and reset the scene graph.

        When a True signal is received, this function:
        1. Waits 5 seconds for the robot to finish current operations
        2. Serializes the current scene graph to a pickle file
        3. Resets the graph to initial state (building + default room)
        4. Increments iteration counter for the next session

        Args:
            msg (std_msgs/Bool): Control signal (True = save and reset)
        """
        
        if msg.data == True:
            # Wait for robot to stabilize
            time.sleep(5.0)

            # Save current graph to pickle file with iteration number
            pickle.dump(
                self.scene_graph, 
                open(f"scene_graph_{self.current_iteration}.pkl", 'wb')
            )

            # Increment iteration counter for next session
            self.current_iteration += 1

            # Reset graph to initial state
            self.scene_graph = nx.Graph()
            self.scene_graph.add_node(0, data=BuildingNode(0, (0.0, 0.0)))
            self.scene_graph.add_node(1, data=RoomNode(1, 'default_room', None, (0.0, 0.0)))
            self.scene_graph.add_edge(0, 1)

            # Reset state variables
            self.graph_lock = False
            self.current_room_id = 1
            self.rooms_classified = False
            self.rooms_segmented = False
            self.rooms = [[1, 999]]
            
    
    def classified_room_callback(self, msg):
        """
        Callback for room classification results.

        Updates the class_id (type) of a room node when its classification
        is received from an external room classification service.

        Args:
            msg (ClassifiedRoom): Contains room ID and classification label
        """
        nodes = list(self.scene_graph.nodes)

        # Find the room node with matching ID and update its label
        for room in self.rooms:
            if self.scene_graph.nodes[room[0]]['data'].id == msg.id.data:
                self.scene_graph.nodes[room[0]]['data'].class_id = msg.label.data
                break
            
    
    def bounding_boxes_callback(self, msg):
        """
        Callback for incoming 3D bounding boxes from object detection.

        This is the main data ingestion point. For each detected object:
        1. Check if it's already in the graph (using overlap threshold)
        2. If new: add as a new node and connect to appropriate room
        3. If existing: merge the new bounding box with the stored one

        The merging approach maintains object identity across multiple
        detections, effectively implementing object tracking.

        Args:
            msg (Object3DBoundingBoxList): List of detected object bounding boxes
        """
        # Prevent processing during initialization
        if len(list(self.scene_graph.nodes)) < 2:
            return

        # Wait for any ongoing graph operations to complete
        while self.graph_lock:
            time.sleep(0.01)

        # Acquire lock for graph modification
        self.graph_lock = True

        # Track which rooms have new objects (for optimization)
        changed_room_ids = []

        # Process each detected object
        for object in msg.bbox:
            # Check if object already exists in graph (by overlap)
            in_graph = self.is_object_in_graph(object)

            # ================================================================
            # CASE 1: New Object - Add to graph
            # ================================================================
            if in_graph == -1:
                # Create new ObjectNode with unique ID
                self.scene_graph.add_node(
                    self.n, 
                    data=ObjectNode(self.n, object.name.data, object.bounding_box)
                )

                # Connect object to a room
                if not self.rooms_classified:
                    # Before room segmentation: add to default room
                    self.scene_graph.add_edge(self.current_room_id, self.n)
                else:
                    # After room segmentation: find containing room
                    nodes = list(self.scene_graph.nodes)
                    for room in self.rooms:
                        # Use point-in-polygon test with object center
                        if self.is_object_in_room(
                            self.get_object_2d_position(object),
                            self.scene_graph.nodes[room[0]]['data'].polygon):

                            # Connect object to this room
                            self.scene_graph.add_edge(
                                self.scene_graph.nodes[room[0]]['data'].id, 
                                self.n
                            )

                            # Track changed room
                            if not self.scene_graph.nodes[room[0]]['data'].id in changed_room_ids:
                                changed_room_ids.append(
                                    self.scene_graph.nodes[room[0]]['data'].id
                                )
                            break

                # Increment node ID counter
                self.n += 1

            # ================================================================
            # CASE 2: Existing Object - Merge bounding boxes
            # ================================================================
            else:
                # Merge the new detection with the existing bounding box
                # This averages positions and expands the box to include both
                merged_box = self.merge_bounding_boxes(
                    self.scene_graph.nodes[in_graph]['data'].bounding_box[0],
                    self.scene_graph.nodes[in_graph]['data'].bounding_box[1],
                    object.bounding_box[0], 
                    object.bounding_box[1]
                )

                # Update stored bounding box
                self.scene_graph.nodes[in_graph]['data'].bounding_box[0] = merged_box[0]
                self.scene_graph.nodes[in_graph]['data'].bounding_box[1] = merged_box[1]

        # Release lock
        self.graph_lock = False
        
    
    def publish_markers(self):
        """
        Publish visualization markers for all nodes and edges in the scene graph.

        Creates and publishes multiple types of RViz markers:
        1. Object bounding boxes (semi-transparent red cubes)
        2. Building marker (blue sphere at height 15m)
        3. Room markers (blue spheres at height 8m)
        4. Line markers showing graph structure/relationships
        5. Text labels for all nodes

        The different heights create a hierarchical 3D visualization where:
        - Building is at the top (z=15)
        - Rooms are in the middle (z=8)
        - Objects are at their actual heights (z=object height)
        """
        # ====================================================================
        # 1. OBJECT BOUNDING BOX MARKERS
        # ====================================================================
        marker_array = MarkerArray()
        bounding_boxes = []
        nodes = list(self.scene_graph.nodes)

        # Collect all object bounding boxes
        for node in nodes:
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is ObjectNode:
                    bounding_boxes.append((
                        self.tuple_from_point(self.scene_graph.nodes[node]['data'].bounding_box[0]),
                        self.tuple_from_point(self.scene_graph.nodes[node]['data'].bounding_box[1])
                    ))

        # Create cube marker for each bounding box
        for i, (point1, point2) in enumerate(bounding_boxes):
            marker = self.create_marker_from_bbox(point1, point2, i)
            marker_array.markers.append(marker)

        # ====================================================================
        # 2. BUILDING MARKER
        # ====================================================================
        building_maker_array = MarkerArray()
        for i, node in enumerate(nodes):
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is BuildingNode:
                    building_maker_array.markers.append(
                        self.create_building_marker(
                            self.scene_graph.nodes[node]['data'].center_point, i
                        )
                    )

        # ====================================================================
        # 3. ROOM MARKERS (with deletion of old markers)
        # ====================================================================

        # Delete old room markers first
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

        # Create new room markers
        room_marker_array = MarkerArray()
        for i, node in enumerate(nodes):
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is RoomNode:
                    room_marker_array.markers.append(
                        self.create_room_marker(
                            self.scene_graph.nodes[node]['data'].polygon, i
                        )
                    )
                    self.old_marker_ids.append(i)

        # ====================================================================
        # 4. LINE MARKERS (showing graph edges/relationships)
        # ====================================================================
        line_marker_array = MarkerArray()
        i = 0

        # Lines from building to each room
        for node in nodes:
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is RoomNode:
                    # Building center at z=15, room center at z=8
                    building_center_point = (
                        self.scene_graph.nodes[0]['data'].center_point[0],
                        self.scene_graph.nodes[0]['data'].center_point[1], 
                        15.0
                    )
                    room_center_point = (
                        self.scene_graph.nodes[node]['data'].center_point[0],
                        self.scene_graph.nodes[node]['data'].center_point[1], 
                        8.0
                    )
                    line_marker_array.markers.append(
                        self.create_line_marker(building_center_point, room_center_point, i)
                    )
                    i += 1

                    # Lines from room to each object in that room
                    target_objects = []

                    # Find all objects connected to this room
                    for edge in list(self.scene_graph.edges):
                        if self.scene_graph.nodes[node]['data'].id == edge[0]:
                            target_objects.append(edge[1])
                        elif self.scene_graph.nodes[node]['data'].id == edge[1]:
                            target_objects.append(edge[0])

                    # Create line to each object
                    for target in target_objects:
                        if target in self.scene_graph and 'data' in self.scene_graph.nodes[target]:
                            # Skip building and room nodes (already handled)
                            if type(self.scene_graph.nodes[target]['data']) == BuildingNode or type(self.scene_graph.nodes[target]['data']) == RoomNode:
                                continue

                            # Calculate object center at its actual height
                            target_center_point = self.calculate_bounding_box_center(
                                self.scene_graph.nodes[target]['data'].bounding_box
                            )
                            target_center_point = (
                                target_center_point.x, 
                                target_center_point.y, 
                                target_center_point.z
                            )

                            line_marker_array.markers.append(
                                self.create_line_marker(
                                    self.scene_graph.nodes[node]['data'].center_point,
                                    target_center_point, 
                                    i
                                )
                            )
                            i += 1

        # Lines between adjacent rooms
        for edge in list(self.scene_graph.edges):
            if type(self.scene_graph.nodes[edge[0]]['data']) == RoomNode and type(self.scene_graph.nodes[edge[1]]['data']) == RoomNode:
                # Both nodes are rooms - draw line at z=8
                room_center_point_1 = (
                    self.scene_graph.nodes[edge[0]]['data'].center_point[0],
                    self.scene_graph.nodes[edge[0]]['data'].center_point[1], 
                    8.0
                )
                room_center_point_2 = (
                    self.scene_graph.nodes[edge[1]]['data'].center_point[0],
                    self.scene_graph.nodes[edge[1]]['data'].center_point[1], 
                    8.0
                )
                line_marker_array.markers.append(
                    self.create_line_marker(room_center_point_1, room_center_point_2, i)
                )
                i += 1

        # ====================================================================
        # 5. TEXT LABEL MARKERS
        # ====================================================================
        text_marker_array = MarkerArray()
        id = 0

        # Create text label for each node
        for node in nodes:
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is BuildingNode:
                    # Building label at z=15
                    position = (
                        self.scene_graph.nodes[node]['data'].center_point[0],
                        self.scene_graph.nodes[node]['data'].center_point[1], 
                        15.0
                    )
                    text_marker_array.markers.append(
                        self.create_text_marker(position, "building", id)
                    )
                    id += 1

                elif type(self.scene_graph.nodes[node]['data']) is RoomNode:
                    # Room label at z=8
                    position = (
                        self.scene_graph.nodes[node]['data'].center_point[0],
                        self.scene_graph.nodes[node]['data'].center_point[1], 
                        8.0
                    )
                    text_marker_array.markers.append(
                        self.create_text_marker(
                            position, 
                            self.scene_graph.nodes[node]['data'].class_id, 
                            id
                        )
                    )
                    id += 1

                elif type(self.scene_graph.nodes[node]['data']) is ObjectNode:
                    # Object label at object height
                    position = self.calculate_bounding_box_center(
                        self.scene_graph.nodes[node]['data'].bounding_box
                    )
                    position = (position.x, position.y, position.z)
                    text_marker_array.markers.append(
                        self.create_text_marker(
                            position, 
                            self.scene_graph.nodes[node]['data'].class_id, 
                            id
                        )
                    )
                    id += 1

        # ====================================================================
        # PUBLISH ALL MARKERS
        # ====================================================================
        self.object_bbox_markers_pub.publish(marker_array)
        # self.objects_pub.publish(graph_objects_msg)
        self.building_markers_pub.publish(building_maker_array)
        self.room_markers_pub.publish(room_marker_array)
        self.line_markers_pub.publish(line_marker_array)
        self.text_markers_pub.publish(text_marker_array)
        
    
    def merge_bounding_boxes(self, min1, max1, min2, max2):
        """
        Merge two axis-aligned 3D bounding boxes into one that encompasses both.

        This is used for object tracking - when the same object is detected
        multiple times, the bounding boxes are merged to get a more accurate
        representation. The merged box is the minimum axis-aligned box that
        contains both input boxes.

        FIXME note: The current implementation simply takes the min/max of
        corners, which works but doesn't handle cases where one detection is
        significantly better than another. A weighted average based on
        confidence scores might be more robust.

        Args:
            min1 (Point32): Minimum corner of first box
            max1 (Point32): Maximum corner of first box
            min2 (Point32): Minimum corner of second box
            max2 (Point32): Maximum corner of second box

        Returns:
            tuple: (merged_min, merged_max) as Point32 objects
        """
        # Convert Point32 to tuples for easier manipulation
        min1 = (min1.x, min1.y, min1.z)
        min2 = (min2.x, min2.y, min2.z)
        max1 = (max1.x, max1.y, max1.z)
        max2 = (max2.x, max2.y, max2.z)

        # Take minimum of minimums and maximum of maximums for each axis
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

        # Convert back to Point32
        merged_min = Point32(x=merged_min[0], y=merged_min[1], z=merged_min[2])
        merged_max = Point32(x=merged_max[0], y=merged_max[1], z=merged_max[2])

        return merged_min, merged_max
    
    
    # ========================================================================
    # MARKER CREATION METHODS
    # ========================================================================

    def create_point_marker_from_bbox(self, point, id):
        """
        Create a SPHERE marker for visualizing a bounding box corner point.

        Used for debugging - shows individual corner points of bounding boxes.

        Args:
            point (tuple): (x, y, z) coordinates
            id (int): Unique marker ID

        Returns:
            visualization_msgs/Marker: Green sphere marker
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "bounding_box_points"
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Position at the point
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = point[2]

        # Small sphere (10cm diameter)
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1

        # Semi-transparent green
        marker.color.a = 0.7
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        return marker


    def create_building_marker(self, center_point, id):
        """
        Create a SPHERE marker representing the building node.

        The building marker is displayed at z=15m for hierarchical visualization,
        making it easily distinguishable from room and object markers.

        Args:
            center_point (tuple): (x, y) coordinates of building center
            id (int): Unique marker ID

        Returns:
            visualization_msgs/Marker: Blue sphere marker at height 15m
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "bounding_box_points"
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Lifetime of 600ms (deleted if not refreshed)
        marker.lifetime = Duration(seconds=0, nanoseconds=600000000).to_msg()

        # Position at z=15 for hierarchical visualization
        marker.pose.position.x = center_point[0]
        marker.pose.position.y = center_point[1]
        marker.pose.position.z = 15.0

        # Identity orientation (not rotated)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # 1m diameter sphere
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        # Blue color (consistent theme for scene graph elements)
        marker.color.a = 1.0
        marker.color.r = 0.25
        marker.color.g = 0.60
        marker.color.b = 0.84

        return marker


    def create_text_marker(self, position, text, id):
        """
        Create a TEXT_VIEW_FACING marker for node labels.

        These text labels always face the camera in RViz, making them
        readable from any viewing angle.

        Args:
            position (tuple): (x, y, z) coordinates
            text (str): Text to display
            id (int): Unique marker ID

        Returns:
            visualization_msgs/Marker: Text marker positioned above the node
        """
        text_marker = Marker()
        text_marker.header.frame_id = "map"
        text_marker.header.stamp = self.get_clock().now().to_msg()
        text_marker.ns = "labels"
        text_marker.id = id
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD

        # Lifetime of 600ms
        text_marker.lifetime = Duration(seconds=0, nanoseconds=600000000).to_msg()

        # Position text 0.8m above the node for visibility
        pos = Point()
        pos.x = position[0]
        pos.y = position[1]
        pos.z = position[2] + 0.8
        text_marker.pose.position = pos

        # Text height in meters
        text_marker.scale.z = 0.7

        # Identity orientation
        text_marker.pose.orientation.x = 0.0
        text_marker.pose.orientation.y = 0.0
        text_marker.pose.orientation.z = 0.0
        text_marker.pose.orientation.w = 1.0

        # Blue color matching other scene graph elements
        text_marker.color.r = 0.25
        text_marker.color.g = 0.60
        text_marker.color.b = 0.84
        text_marker.color.a = 1.0

        # Set the text content
        text_marker.text = text

        return text_marker


    def create_room_marker(self, polygon, id):
        """
        Create a SPHERE marker representing a room node.

        The room marker is placed at the centroid of the room polygon
        at z=8m (between building at z=15 and objects at actual heights).

        Args:
            polygon (list): List of Point32 defining room boundary
            id (int): Unique marker ID

        Returns:
            visualization_msgs/Marker: Blue sphere at room centroid
        """
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "room_markers"
        marker.id = id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        # Lifetime of 600ms
        marker.lifetime = Duration(seconds=0, nanoseconds=600000000).to_msg()

        # Calculate centroid of room polygon
        center_x = 0.0
        center_y = 0.0
        if polygon is not None:
            for point in polygon:
                center_x += point.x
                center_y += point.y
            center_x /= len(polygon)
            center_y /= len(polygon)

        # Position at z=8 for hierarchical visualization
        marker.pose.position.x = center_x
        marker.pose.position.y = center_y
        marker.pose.position.z = 8.0

        # Identity orientation
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        # 1m diameter sphere
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        # Blue color
        marker.color.a = 1.0
        marker.color.r = 0.25
        marker.color.g = 0.60
        marker.color.b = 0.84

        return marker


    def create_line_marker(self, point1, point2, i):
        """
        Create a LINE_STRIP marker connecting two nodes in the graph.

        Lines visualize the graph structure - edges represent relationships:
        - Building to rooms
        - Rooms to objects
        - Adjacent rooms to each other

        Args:
            point1 (tuple): (x, y, z) start point
            point2 (tuple): (x, y, z) end point
            i (int): Unique marker ID

        Returns:
            visualization_msgs/Marker: Blue line connecting the points
        """
        # Extract z coordinates (default to 8.0 if not provided)
        z1 = 8.0
        z2 = 8.0
        if len(point1) > 2:
            z1 = point1[2]
        if len(point2) > 2:
            z2 = point2[2]

        # Convert to Point messages
        point1 = Point(x=point1[0], y=point1[1], z=z1)
        point2 = Point(x=point2[0], y=point2[1], z=z2)

        line_marker = Marker()
        line_marker.header.frame_id = "map"
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = "lines"
        line_marker.id = i
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD

        # Line width (4cm)
        line_marker.scale.x = 0.04

        # Lifetime of 600ms
        line_marker.lifetime = Duration(seconds=0, nanoseconds=600000000).to_msg()

        # Semi-transparent blue
        line_marker.color.r = 0.25
        line_marker.color.g = 0.60
        line_marker.color.b = 0.84
        line_marker.color.a = 0.7

        # Identity orientation
        line_marker.pose.orientation.x = 0.0
        line_marker.pose.orientation.y = 0.0
        line_marker.pose.orientation.z = 0.0
        line_marker.pose.orientation.w = 1.0

        # Add the two endpoints
        line_marker.points.append(point1)
        line_marker.points.append(point2)

        return line_marker

    def create_marker_from_bbox(self, point1, point2, id):
        """
        Create a CUBE marker from a 3D bounding box.

        This creates a semi-transparent cube visualization of an object's
        bounding box. The cube is positioned at the box center with
        dimensions matching the box size.

        The function computes:
        1. Center point (average of corners)
        2. Dimensions (differences along each axis)
        3. Orientation (from axis-aligned vectors)
        4. Quaternion representation of orientation

        Args:
            point1 (tuple): (x, y, z) minimum corner
            point2 (tuple): (x, y, z) maximum corner
            id (int): Unique marker ID

        Returns:
            visualization_msgs/Marker: Semi-transparent red cube
        """
        marker = Marker()

        # Calculate center point
        center_x = (point1[0] + point2[0]) / 2.0
        center_y = (point1[1] + point2[1]) / 2.0
        center_z = (point1[2] + point2[2]) / 2.0

        # Calculate dimensions (extents along each axis)
        size_x = abs(point1[0] - point2[0])
        size_y = abs(point1[1] - point2[1])
        size_z = abs(point1[2] - point2[2])

        # Create orthonormal basis from bounding box axes
        # Note: For axis-aligned boxes, this just gives identity rotation
        v1 = np.array([point2[0] - point1[0], 0, 0])
        v2 = np.array([0, point2[1] - point1[1], 0])
        v3 = np.array([0, 0, point2[2] - point1[2]])

        # Normalize vectors to unit length
        x_axis = v1 / np.linalg.norm(v1)
        y_axis = v2 / np.linalg.norm(v2)
        z_axis = v3 / np.linalg.norm(v3)

        # Construct 3x3 rotation matrix from basis vectors
        rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

        # Create 4x4 homogeneous transformation matrix (not used here)
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix

        # Convert rotation matrix to quaternion using scipy
        rotation_obj = R.from_matrix(rotation_matrix)
        quaternion = rotation_obj.as_quat()  # Returns [x, y, z, w]

        # Fill in marker properties
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "bounding_boxes"
        marker.id = id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Lifetime of 600ms
        marker.lifetime = Duration(seconds=0, nanoseconds=600000000).to_msg()

        # Set position to center of bounding box
        marker.pose.position.x = center_x
        marker.pose.position.y = center_y
        marker.pose.position.z = center_z

        # Set orientation from quaternion
        marker.pose.orientation.x = quaternion[0]
        marker.pose.orientation.y = quaternion[1]
        marker.pose.orientation.z = quaternion[2]
        marker.pose.orientation.w = quaternion[3]

        # Set scale to bounding box dimensions
        marker.scale.x = size_x
        marker.scale.y = size_y
        marker.scale.z = size_z

        # Semi-transparent red for object bounding boxes
        marker.color.a = 0.5
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        return marker
    
    
    def rooms_callback(self, msg):
        """
        Process incoming room segmentation data and reorganize the scene graph.

        This callback is triggered when room boundaries are received from a room
        segmentation algorithm. It performs a major graph restructuring:

        1. Removes all existing room nodes (except default room)
        2. Creates new room nodes from the segmentation polygons
        3. Calculates room centroids and areas
        4. Recomputes building center as average of all room centers
        5. Reassigns all objects to their containing rooms based on geometry

        The transition from a single default room to multiple segmented rooms
        enables more accurate spatial reasoning and room-level queries.

        Args:
            msg (RoomPolygonList): Contains list of room polygons with vertex points
        """
        # Wait for any ongoing graph operations to complete
        while self.graph_lock:
            time.sleep(0.01)

        # Acquire lock to prevent concurrent modifications
        self.graph_lock = True

        # ========================================================================
        # STEP 1: Remove existing room nodes
        # ========================================================================
        for node in list(self.scene_graph.nodes):
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is RoomNode:
                    # Remove old room node and its edges
                    self.scene_graph.remove_node(node)

        # Reset room tracking structures
        self.rooms = []
        self.rooms_classified = True  # Mark that room segmentation is complete

        # Remove all edges (will be reconstructed based on new room structure)
        self.scene_graph.remove_edges_from(list(self.scene_graph.edges))

        # ========================================================================
        # STEP 2: Create new room nodes from segmentation
        # ========================================================================
        for room in msg.rooms:
            # Calculate room centroid by averaging all polygon vertices
            center_x = 0.0
            center_y = 0.0

            if room.points is not None:
                for point in room.points:
                    center_x += point.x
                    center_y += point.y
                center_x /= len(room.points)
                center_y /= len(room.points)

            # Create new RoomNode with:
            # - Unique ID (self.n)
            # - Placeholder class_id '-' (will be updated by classifier)
            # - Polygon boundary points
            # - Computed centroid
            self.scene_graph.add_node(
                self.n, 
                data=RoomNode(self.n, '-', room.points, (center_x, center_y))
            )

            # Connect room to building root (node 0)
            self.scene_graph.add_edge(0, self.n)

            # Store [room_id, room_area] for sorting
            # Area is used to determine which room the robot starts in
            # (assumption: robot typically starts in largest room)
            self.rooms.append([self.n, self.polygon_area(room.points)])

            # Increment node ID counter
            self.n += 1

        # ========================================================================
        # STEP 3: Sort rooms by area (largest first)
        # ========================================================================
        # This ordering helps with room selection heuristics
        self.rooms = sorted(self.rooms, key=lambda x: x[1])

        # ========================================================================
        # STEP 4: Recalculate building center
        # ========================================================================
        # Building center is the average of all room centroids
        # This provides a geometric center of the entire structure
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

        # Update building node center
        self.scene_graph.nodes[0]['data'].center_point = (center_x, center_y)

        # ========================================================================
        # STEP 5: Reassign all objects to appropriate rooms
        # ========================================================================
        # For each object, find its nearest room (fallback strategy)
        # Ideally would use point-in-polygon test, but this uses distance

        for object_node in nodes:
            if object_node in self.scene_graph and 'data' in self.scene_graph.nodes[object_node]:
                if type(self.scene_graph.nodes[object_node]['data']) is ObjectNode:
                    min_dist = np.inf
                    min_node = 0

                    # Find the closest room to this object
                    for room_node in nodes:
                        if room_node in self.scene_graph and 'data' in self.scene_graph.nodes[room_node]:
                            if type(self.scene_graph.nodes[room_node]['data']) is RoomNode:
                                # Calculate 2D Euclidean distance between:
                                # - Object center (from bounding box)
                                # - Room centroid
                                dist = self.euclidean_distance2d(
                                    self.get_object_2d_position(self.scene_graph.nodes[object_node]['data']),
                                    self.scene_graph.nodes[room_node]['data'].center_point
                                )

                                if dist < min_dist:
                                    min_dist = dist
                                    min_node = room_node

                    # Create edge connecting object to nearest room
                    self.scene_graph.add_edge(min_node, object_node)

        # Mark that room segmentation has been processed
        self.rooms_segmented = True

        # Release lock
        self.graph_lock = False
    
    
    def get_object_2d_position(self, object):
        """
        Extract the 2D center position of an object from its bounding box.

        Handles two input formats:
        1. ObjectNode with .bounding_box attribute
        2. Raw bounding box list [min_corner, max_corner]

        Args:
            object: ObjectNode or bounding box list

        Returns:
            tuple: (center_x, center_y) in 2D space
        """
        if hasattr(object, 'bounding_box'):
            # Object is an ObjectNode - access bounding_box attribute
            center_x = (object.bounding_box[0].x + object.bounding_box[1].x) / 2
            center_y = (object.bounding_box[0].y + object.bounding_box[1].y) / 2
        else:
            # Object is a raw bounding box list
            center_x = (object[0].x + object[1].x) / 2
            center_y = (object[0].y + object[1].y) / 2

        return (center_x, center_y)


    def is_object_in_graph(self, object):
        """
        Check if a detected object already exists in the scene graph.

        Uses overlap-based association with IoU (Intersection over Union):
        - Calculates overlap volume between new detection and all existing objects
        - If overlap/volume ratio > threshold (0.8), objects are considered the same
        - Tests ratio in both directions (prevents small object absorption)

        This implements object tracking - maintaining object identity across
        multiple detections even with slight position/size variations.

        Args:
            object (Object3DBoundingBox): New detected object with bounding box

        Returns:
            int: Node ID if object exists in graph, -1 if new object
        """
        nodes = list(self.scene_graph.nodes)

        # Calculate center of new detection for logging/debugging
        center_point = self.calculate_bounding_box_center(object.bounding_box)
        center_point = [center_point.x, center_point.y, center_point.z]

        # Check against all existing object nodes
        for node in nodes:
            if node in self.scene_graph and 'data' in self.scene_graph.nodes[node]:
                if type(self.scene_graph.nodes[node]['data']) is ObjectNode:

                    # Calculate intersection volume
                    overlap_volume = self.overlap_volume(
                        object.bounding_box,
                        self.scene_graph.nodes[node]['data'].bounding_box
                    )

                    # Calculate IoU ratios in both directions
                    # Ratio 1: overlap / new_object_volume
                    # Ratio 2: overlap / existing_object_volume
                    # Using OR ensures we catch partial overlaps in either direction

                    if (overlap_volume / self.bounding_box_volume(object.bounding_box) > self.overlapping_threshold or
                        overlap_volume / self.bounding_box_volume(self.scene_graph.nodes[node]['data'].bounding_box) > self.overlapping_threshold):
                        # Objects overlap significantly - same object
                        return node

        # No matching object found - this is a new detection
        return -1


    def is_point_inside_bbox(self, point1, point2, P):
        """
        Check if a 3D point lies inside an axis-aligned bounding box.

        Simple containment test: point must be within min/max bounds
        along all three axes simultaneously.

        Args:
            point1 (Point32): First corner of bounding box
            point2 (Point32): Opposite corner of bounding box
            P (Point32): Point to test

        Returns:
            bool: True if point is inside bounding box, False otherwise
        """
        # Extract coordinates
        x1, y1, z1 = point1.x, point1.y, point1.z
        x2, y2, z2 = point2.x, point2.y, point2.z
        px, py, pz = P.x, P.y, P.z

        # Check if point is within bounds on each axis
        # Use min/max to handle either corner ordering
        inside_x = min(x1, x2) <= px <= max(x1, x2)
        inside_y = min(y1, y2) <= py <= max(y1, y2)
        inside_z = min(z1, z2) <= pz <= max(z1, z2)

        # Point is inside only if it's within bounds on ALL axes
        return inside_x and inside_y and inside_z

    
    def calculate_bounding_box_center(self, bounding_box):
        """
        Calculate the 3D center point of an axis-aligned bounding box.

        This method computes the centroid by averaging the two corner points
        (min and max). For an axis-aligned bounding box (AABB), the center is
        simply the midpoint between the minimum and maximum corners.

        Mathematical formula:
            center = (min_corner + max_corner) / 2

        Args:
            bounding_box (list): List containing two Point32 objects:
                                [0] = minimum corner (smallest x, y, z)
                                [1] = maximum corner (largest x, y, z)

        Returns:
            Point32: Center point with x, y, z coordinates

        Note: This assumes the bounding box list has exactly 2 points.
            The division by 2 is mathematically correct because we're
            summing exactly 2 corner points.
        """
        # Initialize accumulators for each coordinate
        sum_x = sum_y = sum_z = 0.0

        # Sum the coordinates of both corner points
        for point in bounding_box:
            sum_x += point.x
            sum_y += point.y
            sum_z += point.z

        # Calculate center by dividing by 2 (since we have 2 points)
        center_x = sum_x / 2
        center_y = sum_y / 2
        center_z = sum_z / 2

        # Return as Point32 message type for ROS compatibility
        return Point32(x=center_x, y=center_y, z=center_z)


    def euclidean_distance(self, point_1, point_2):
        """
        Calculate the 3D Euclidean distance between two points.

        Uses the standard Euclidean distance formula in 3D space:
            d = sqrt((x₂-x₁)² + (y₂-y₁)² + (z₂-z₁)²)

        This is the straight-line distance between two points in 3D space,
        useful for proximity calculations, nearest-neighbor searches, and
        spatial relationship queries.

        Args:
            point_1 (Point32): First point with x, y, z coordinates
            point_2 (Point32): Second point with x, y, z coordinates

        Returns:
            float: Euclidean distance in meters (or whatever units the coordinates use)

        Computational complexity: O(1) - constant time
        """
        return math.sqrt((point_2.x - point_1.x)**2 + 
                        (point_2.y - point_1.y)**2 + 
                        (point_2.z - point_1.z)**2)


    def euclidean_distance2d(self, point_1, point_2):
        """
        Calculate the 2D Euclidean distance between two points (ignoring height).

        This projects the distance calculation onto the XY plane, effectively
        computing horizontal distance while ignoring vertical (z) differences.

        Formula:
            d = sqrt((x₂-x₁)² + (y₂-y₁)²)

        Use cases:
        - Room-to-object association (height is less relevant)
        - Floor-level navigation distance
        - Checking if rooms are adjacent (based on wall proximity)

        Args:
            point_1 (tuple): First point as (x, y) or (x, y, z) - only x, y used
            point_2 (tuple): Second point as (x, y) or (x, y, z) - only x, y used

        Returns:
            float: 2D Euclidean distance in meters

        Note: Unlike euclidean_distance(), this expects tuple input rather than
            Point32 messages. This is for compatibility with room centroid
            representations which are stored as (x, y) tuples.
        """
        return math.sqrt((point_2[0] - point_1[0])**2 + 
                        (point_2[1] - point_1[1])**2)


    def tuple_from_point(self, point):
        """
        Convert a ROS Point32 message to a Python tuple.

        This is a convenience function for interfacing between ROS message types
        and standard Python data structures. Many NumPy and geometric algorithms
        work better with tuples/arrays than with ROS message objects.

        Args:
            point (Point32): ROS message with x, y, z attributes

        Returns:
            tuple: (x, y, z) as a 3-element tuple of floats
        """
        return (point.x, point.y, point.z)


    def calculate_center(self, corners):
        """
        Calculate the center of a bounding box from its 8 corner points.

        This is an alternative center calculation method when all 8 corners of
        a 3D bounding box are available (not just min/max). The center is computed
        as the mean (average) position of all corner points.

        For a perfect box, this should give the same result as averaging min/max,
        but this method is more robust to slight irregularities or rotated boxes.

        Args:
            corners (list): List of 8 corner points, each as a tuple (x, y, z)
                        Typically ordered as:
                        [0]: (min_x, min_y, min_z) - front-left-bottom
                        [1]: (max_x, min_y, min_z) - front-right-bottom
                        [2]: (min_x, max_y, min_z) - back-left-bottom
                        [3]: (max_x, max_y, min_z) - back-right-bottom
                        [4]: (min_x, min_y, max_z) - front-left-top
                        [5]: (max_x, min_y, max_z) - front-right-top
                        [6]: (min_x, max_y, max_z) - back-left-top
                        [7]: (max_x, max_y, max_z) - back-right-top

        Returns:
            tuple: (center_x, center_y, center_z) as a 3-element tuple
        """
        # Convert list of tuples to NumPy array for efficient computation
        corners_np = np.array(corners)

        # Calculate mean along axis 0 (averages all x's, all y's, all z's)
        # axis=0 means "collapse rows" → one value per column
        center = np.mean(corners_np, axis=0)

        # Convert back to tuple for consistency with other methods
        return tuple(center)


    def calculate_orientation(self, corners):
        """
        Calculate the orientation (rotation) of a bounding box from corner points.

        This extracts the 3D rotation of a bounding box by computing three
        orthonormal edge vectors that define its local coordinate frame. These
        vectors are then converted to a quaternion representation.

        Algorithm:
        1. Compute three edge vectors from corner[0] to corners[1,3,4]
        2. Normalize these vectors to unit length (defines orthonormal basis)
        3. Construct a 3×3 rotation matrix from the basis vectors
        4. Convert rotation matrix to quaternion using scipy
        5. Normalize the quaternion to ensure unit magnitude

        The quaternion represents the rotation that would transform the world
        coordinate frame to align with the bounding box's local frame.

        Args:
            corners (list): List of 8 corner points (see calculate_center for order)
                        Corners[0,1,3,4] must form three perpendicular edges

        Returns:
            tuple: Normalized quaternion (x, y, z, w) representing orientation

        Mathematical background:
            - Quaternions are a 4D representation of 3D rotations: q = w + xi + yj + zk
            - They avoid gimbal lock and interpolate smoothly (unlike Euler angles)
            - A unit quaternion (|q| = 1) represents a valid rotation
            - Scipy's as_quat() returns [x, y, z, w] order (Hamilton convention)
        """
        # Calculate three edge vectors from origin corner (corner[0])
        # These define the X, Y, Z axes of the box's local coordinate frame
        vec_x = np.array(corners[1]) - np.array(corners[0])  # X-axis direction
        vec_y = np.array(corners[3]) - np.array(corners[0])  # Y-axis direction
        vec_z = np.array(corners[4]) - np.array(corners[0])  # Z-axis direction

        # Normalize vectors to unit length (required for orthonormal basis)
        # This ensures the rotation matrix represents pure rotation, no scaling
        vec_x /= np.linalg.norm(vec_x)
        vec_y /= np.linalg.norm(vec_y)
        vec_z /= np.linalg.norm(vec_z)

        # Construct 4×4 homogeneous transformation matrix (with identity initialization)
        # Using 4×4 for potential future use with translations, but only 3×3 rotation needed
        rotation_matrix = np.identity(4)

        # Fill in the 3×3 rotation submatrix with normalized basis vectors
        # Each column represents one basis vector of the rotated frame
        rotation_matrix[0:3, 0] = vec_x  # First column: X-axis
        rotation_matrix[0:3, 1] = vec_y  # Second column: Y-axis
        rotation_matrix[0:3, 2] = vec_z  # Third column: Z-axis

        # Convert the 3×3 rotation matrix to a quaternion using scipy
        # Extract only the rotation part (upper-left 3×3)
        rotation_obj = R.from_matrix(rotation_matrix[:3, :3])
        quaternion = rotation_obj.as_quat()  # Returns [x, y, z, w]

        # Normalize to ensure unit quaternion (required for valid rotation)
        return self.normalize_quaternion(quaternion)


    def normalize_quaternion(self, quat):
        """
        Normalize a quaternion to unit length.

        A valid rotation quaternion must satisfy |q| = sqrt(x² + y² + z² + w²) = 1.
        This function enforces that constraint by dividing all components by the
        quaternion's magnitude.

        Normalization is necessary because:
        1. Floating-point arithmetic can introduce small errors
        2. Interpolated quaternions may drift from unit magnitude
        3. Non-unit quaternions don't represent pure rotations

        Args:
            quat (tuple/array): Quaternion as (x, y, z, w) - may not be normalized

        Returns:
            tuple: Normalized quaternion (x, y, z, w) with magnitude 1.0

        Special case:
            If the input quaternion has zero magnitude (degenerate case), returns
            the identity quaternion (0, 0, 0, 1) which represents no rotation.

        Mathematical formula:
            q_normalized = q / |q| where |q| = sqrt(x² + y² + z² + w²)
        """
        # Calculate the magnitude (Euclidean norm) of the quaternion
        norm = np.linalg.norm(quat)

        # Handle degenerate case: zero-magnitude quaternion
        if norm == 0:
            self.get_logger().warn("Zero norm quaternion, cannot normalize!")
            return (0, 0, 0, 1)  # Return identity quaternion (no rotation)

        # Normalize by dividing all components by the magnitude
        # Convert to tuple for consistency with other return types
        return tuple(np.array(quat) / norm)


    def calculate_scale(self, corners):
        """
        Calculate the dimensions (scale) of a bounding box from corner points.

        Computes the physical size of the bounding box along each axis by
        measuring the distance between adjacent corners. These dimensions
        represent the length, width, and height of the object.

        Args:
            corners (list): List of 8 corner points (see calculate_center for order)

        Returns:
            tuple: (scale_x, scale_y, scale_z) dimensions in meters
                scale_x: length along X-axis (left-right)
                scale_y: length along Y-axis (forward-backward)
                scale_z: length along Z-axis (up-down / height)

        Computation:
            - scale_x = distance from corner[0] to corner[1]
            - scale_y = distance from corner[0] to corner[3]
            - scale_z = distance from corner[0] to corner[4]

        These scales can be used for:
        - Volume calculation: volume = scale_x × scale_y × scale_z
        - Visualization marker sizing in RViz
        - Collision detection and physics simulation
        """
        # Calculate length along X-axis (corner 0 to corner 1)
        scale_x = np.linalg.norm(np.array(corners[1]) - np.array(corners[0]))

        # Calculate length along Y-axis (corner 0 to corner 3)
        scale_y = np.linalg.norm(np.array(corners[3]) - np.array(corners[0]))

        # Calculate length along Z-axis (corner 0 to corner 4)
        scale_z = np.linalg.norm(np.array(corners[4]) - np.array(corners[0]))

        return (scale_x, scale_y, scale_z)


    def bounding_box_volume(self, bounding_box):
        """
        Calculate the volume of a 3D axis-aligned bounding box.

        For an axis-aligned bounding box (AABB), the volume is simply the
        product of its three dimensions. This is equivalent to calculating
        the volume of a rectangular prism.

        Formula:
            V = length_x × length_y × length_z
            V = (x_max - x_min) × (y_max - y_min) × (z_max - z_min)

        Args:
            bounding_box (list): Two-element list containing:
                                [0] = minimum corner Point32
                                [1] = maximum corner Point32

        Returns:
            float: Volume in cubic meters (or cubic units of input coordinates)

        Use cases:
        - Object size classification (large vs. small objects)
        - Intersection-over-Union (IoU) calculations for object tracking
        - Volume-based filtering (remove objects that are too small/large)
        - Physics simulation (mass estimation from volume × density)
        """
        # Extract corner coordinates as lists for easier manipulation
        min_point = [bounding_box[0].x, bounding_box[0].y, bounding_box[0].z]
        max_point = [bounding_box[1].x, bounding_box[1].y, bounding_box[1].z]

        # Calculate dimensions along each axis
        length_x = max_point[0] - min_point[0]  # X-axis extent
        length_y = max_point[1] - min_point[1]  # Y-axis extent
        length_z = max_point[2] - min_point[2]  # Z-axis extent

        # Volume is the product of the three dimensions
        volume = length_x * length_y * length_z

        return volume


    def overlap_volume(self, bounding_box1, bounding_box2):
        """
        Calculate the volume of intersection between two 3D bounding boxes.

        This is a crucial function for object tracking and association. It computes
        the 3D Intersection over Union (IoU) metric by finding the overlapping
        region between two axis-aligned bounding boxes.

        Algorithm:
        1. For each axis (x, y, z):
        - Find the overlap interval: [max(min1, min2), min(max1, max2)]
        - If overlap_start > overlap_end, there's no overlap (set to 0)
        2. Multiply the three overlap lengths to get intersection volume

        Mathematical formula:
            overlap_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            overlap_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            overlap_z = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
            overlap_volume = overlap_x × overlap_y × overlap_z

        Args:
            bounding_box1 (list): [min_corner, max_corner] for first box
            bounding_box2 (list): [min_corner, max_corner] for second box

        Returns:
            float: Intersection volume in cubic meters
                Returns 0.0 if boxes don't overlap at all

        Use cases:
        - Object tracking: match detections across frames (high overlap = same object)
        - Collision detection: check if two objects intersect
        - IoU calculation: IoU = overlap_volume / (volume1 + volume2 - overlap_volume)

        Performance: O(1) constant time - just arithmetic operations
        """
        # Extract corner coordinates for both bounding boxes
        box1_min = [bounding_box1[0].x, bounding_box1[0].y, bounding_box1[0].z]
        box1_max = [bounding_box1[1].x, bounding_box1[1].y, bounding_box1[1].z]
        box2_min = [bounding_box2[0].x, bounding_box2[0].y, bounding_box2[0].z]
        box2_max = [bounding_box2[1].x, bounding_box2[1].y, bounding_box2[1].z]

        # Calculate overlap along X-axis
        # Overlap = min of max values - max of min values
        # If result is negative, there's no overlap → max(0, ...) ensures non-negative
        x_overlap = max(0, min(box1_max[0], box2_max[0]) - max(box1_min[0], box2_min[0]))

        # Calculate overlap along Y-axis
        y_overlap = max(0, min(box1_max[1], box2_max[1]) - max(box1_min[1], box2_min[1]))

        # Calculate overlap along Z-axis
        z_overlap = max(0, min(box1_max[2], box2_max[2]) - max(box1_min[2], box2_min[2]))

        # Calculate intersection volume as product of overlaps
        # If any overlap is 0, the entire volume will be 0 (no intersection)
        overlap_vol = x_overlap * y_overlap * z_overlap

        return overlap_vol


    def polygon_area(self, vertices):
        """
        Calculate the area of a 2D polygon using the Shoelace formula.

        The Shoelace formula (also called surveyor's formula or Gauss's area formula)
        computes polygon area directly from vertex coordinates without triangulation.

        Formula:
            Area = (1/2) × |Σ(x[i] × y[i+1] - x[i+1] × y[i])| for i = 0 to n-1

        Geometric interpretation:
            The formula sums the signed areas of trapezoids formed by projecting
            each polygon edge onto the x-axis. Counterclockwise edges contribute
            positive area, clockwise edges contribute negative area.

        Properties:
        - Works for any simple polygon (convex or concave)
        - Automatically handles vertex ordering (CW or CCW)
        - Time complexity: O(n) where n = number of vertices
        - Numerically stable for typical coordinate ranges

        Args:
            vertices (list): List of Point32 objects defining polygon boundary
                            Vertices should be ordered consecutively around perimeter
                            (either clockwise or counterclockwise)

        Returns:
            float: Polygon area in square meters (always positive)

        Use cases:
        - Room size calculation and sorting (larger rooms first)
        - Determining which room the robot starts in
        - Spatial queries (object density per room area)
        """
        n = len(vertices)
        area = 0

        # Iterate through all edges of the polygon
        # Each edge contributes a signed trapezoid area to the total
        for i in range(n):
            x1 = vertices[i].x
            y1 = vertices[i].y
            # Use modulo to wrap from last vertex back to first
            x2 = vertices[(i + 1) % n].x
            y2 = vertices[(i + 1) % n].y

            # Add the cross product (signed area contribution)
            # Positive for counterclockwise edges, negative for clockwise
            area += x1 * y2 - y1 * x2

        # Take absolute value and divide by 2 to get final area
        # Absolute value ensures positive result regardless of vertex ordering
        return abs(area) / 2


    def has_adjacent_points(self, polygon_1, polygon_2):
        """
        Check if two polygons have vertices that are close to each other.

        This determines if two rooms are adjacent (share a boundary or wall).
        The algorithm uses a sparse sampling approach to balance accuracy and
        computational efficiency.

        Algorithm:
        1. Sample every 10th vertex from each polygon (reduces O(n²) to O(n²/100))
        2. For each sampled vertex pair, calculate 2D distance
        3. If any distance < 0.5 meters, consider rooms adjacent

        Design decisions:
        - 0.5m threshold: Allows for small gaps from sensor noise or wall thickness
        - Every 10th point: Trades some accuracy for 100× speedup
        - 2D distance: Ignores height (rooms on different floors won't be adjacent)

        Limitations:
        - May miss adjacency if sampled points don't align
        - Could give false positives if rooms are very close but not touching
        - Assumes uniform vertex spacing (dense sampling near features)

        Args:
            polygon_1 (list): Vertices of first room polygon
            polygon_2 (list): Vertices of second room polygon

        Returns:
            bool: True if rooms have close points (likely adjacent), False otherwise

        Performance:
            - Best case: O(1) if first points are close
            - Worst case: O((n/10) × (m/10)) = O(nm/100) where n,m are vertex counts
            - Typical: ~100× faster than checking all pairs
        """
        # Sample every 10th vertex from polygon_1 (stride of 10)
        for i in range(0, len(polygon_1), 10):
            # Sample every 10th vertex from polygon_2
            for j in range(0, len(polygon_2), 10):
                # Extract 3D coordinates (z included but not used in distance calc)
                point_1 = (polygon_1[i].x, polygon_1[i].y, polygon_1[i].z)
                point_2 = (polygon_2[j].x, polygon_2[j].y, polygon_2[j].z)

                # Check 2D distance (ignoring height difference)
                # 0.5m threshold accounts for sensor noise and wall thickness
                if self.euclidean_distance2d(point_1, point_2) < 0.5:
                    return True  # Found adjacent points - rooms are neighbors

        # No close points found - rooms are not adjacent
        return False


    def export_scene_graph_to_json(self, filename):
        """
        Export the current scene graph to a hierarchical JSON file.

        This function serializes the entire scene graph into a structured JSON
        representation suitable for:
        - Persistent storage and replay of captured environments
        - Visualization in external 3D viewers or web interfaces
        - Integration with other robotic systems and databases
        - Scene understanding and semantic reasoning algorithms
        - Dataset creation for machine learning

        JSON structure hierarchy:
            Scene (root)
            └── Building
                └── Floors
                    └── Rooms
                        └── Objects

        Each level contains:
        - Unique identifiers (id)
        - Parent relationship (parent_id)
        - Semantic labels (label, semantic_label)
        - Spatial information (centroide, bounding_box)
        - Confidence scores (for objects)

        Args:
            filename (str): Output JSON file path (e.g., "scene_graph.json")

        Output format example:
        {
        "scene_id": "generated_scene",
        "method_name": "GraphManagementNode_Export",
        "building": {
            "id": "building_0",
            "floors": [{
            "id": "floor_0",
            "rooms": [{
                "id": "room_1",
                "label": "kitchen",
                "centroide": {"x": 5.2, "y": 3.1, "z": 0.0},
                "objects": [{
                "id": "obj_10",
                "semantic_label": "chair",
                "confidence": 1.0,
                "centroide": {"x": 5.5, "y": 3.0, "z": 0.5},
                "bounding_box": {...}
                }]
            }]
            }]
        }
        }
        """
        # ========================================================================
        # STEP 1: Find the building node (root of the scene graph)
        # ========================================================================
        building_node = None
        for node in self.scene_graph.nodes:
            data = self.scene_graph.nodes[node].get('data')
            if isinstance(data, BuildingNode):
                building_node = data
                break

        # Create building object with placeholder bounding box
        # (Could be computed as the extent of all rooms if needed)
        building = {
            "id": f"building_{building_node.id if building_node else '-'}",
            "label": "-",  # Placeholder - could be building name/type
            "bounding_box": {
                "min_corner": {"x": 0.0, "y": 0.0, "z": 0.0},
                "max_corner": {"x": 0.0, "y": 0.0, "z": 0.0}
            },
            "floors": []  # Will be populated below
        }

        # ========================================================================
        # STEP 2: Create floor object (currently assumes single floor)
        # ========================================================================
        # Future enhancement: Could support multi-floor buildings by grouping
        # rooms based on z-coordinate or explicit floor assignments
        floor = {
            "id": "floor_0",
            "parent_id": building["id"],
            "label": "-",  # Could be "ground floor", "first floor", etc.
            "centroide": {
                "x": building_node.center_point[0],
                "y": building_node.center_point[1],
                "z": 0.0  # Floor level height
            },
            "bounding_box": {
                "min_corner": {"x": 0.0, "y": 0.0, "z": 0.0},
                "max_corner": {"x": 0.0, "y": 0.0, "z": 0.0}
            },
            "rooms": []  # Will be populated in next step
        }

        # ========================================================================
        # STEP 3: Iterate through all room nodes and their objects
        # ========================================================================
        for node in self.scene_graph.nodes:
            data = self.scene_graph.nodes[node].get('data')

            # Process only RoomNode types
            if isinstance(data, RoomNode):
                # Create room object with metadata
                room = {
                    "id": f"room_{data.id}",
                    "parent_id": floor["id"],
                    "label": data.class_id,  # Room type: "kitchen", "bedroom", etc.
                    "centroide": {
                        "x": data.center_point[0],
                        "y": data.center_point[1],
                        "z": 0.0  # Room center at floor level
                    },
                    "bounding_box": {
                        # Placeholder - could compute from room polygon extents
                        "min_corner": {"x": 0.0, "y": 0.0, "z": 0.0},
                        "max_corner": {"x": 0.0, "y": 0.0, "z": 0.0}
                    },
                    "objects": []  # Will be populated with room's objects
                }

                # ================================================================
                # STEP 3.1: Find all objects connected to this room
                # ================================================================
                # Iterate through all neighbors of this room node in the graph
                for neighbor in self.scene_graph.neighbors(node):
                    obj_data = self.scene_graph.nodes[neighbor].get('data')

                    # Process only ObjectNode types (skip building/room neighbors)
                    if isinstance(obj_data, ObjectNode):
                        # Calculate object center from bounding box
                        obj_center_x = (obj_data.bounding_box[0].x + obj_data.bounding_box[1].x) / 2
                        obj_center_y = (obj_data.bounding_box[0].y + obj_data.bounding_box[1].y) / 2
                        obj_center_z = (obj_data.bounding_box[0].z + obj_data.bounding_box[1].z) / 2

                        # Create object entry with full spatial information
                        obj = {
                            "id": f"obj_{obj_data.id}",
                            "parent_id": room["id"],
                            "semantic_label": obj_data.class_id,  # "chair", "table", etc.
                            "confidence": 1.0,  # Placeholder - could track detection confidence
                            "centroide": {
                                "x": obj_center_x,
                                "y": obj_center_y,
                                "z": obj_center_z
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
                            "attributes": []  # Placeholder for future attributes
                                            # (color, material, state, etc.)
                        }

                        # Add object to room's object list
                        room["objects"].append(obj)

                # Add room to floor's room list
                floor["rooms"].append(room)

        # Add floor to building's floor list
        building["floors"].append(floor)

        # ========================================================================
        # STEP 4: Create final JSON structure with metadata
        # ========================================================================
        scene_graph_json = {
            "scene_id": "generated_scene",  # Could be timestamp or unique identifier
            "method_name": "GraphManagementNode_Export",  # Identifies export method
            "building": building
        }

        # ========================================================================
        # STEP 5: Write JSON to file with pretty printing
        # ========================================================================
        with open(filename, "w") as f:
            json.dump(scene_graph_json, f, indent=2)  # indent=2 for readability

        # Log successful export
        self.get_logger().info(f"Scene graph exported to {filename}")


    # ============================================================================
    # MAIN ENTRY POINT
    # ============================================================================

def main(args=None):
    """
    Main entry point for the ROS2 scene graph fusion node.

    This function follows the standard ROS2 node lifecycle:
    1. Initialize the ROS2 Python client library
    2. Create the node instance
    3. Enter the event loop (spin) to process callbacks
    4. Handle graceful shutdown on interruption
    5. Clean up resources

    The spin() call blocks until the node is shut down via:
    - Ctrl+C (KeyboardInterrupt)
    - Kill signal from OS
    - Explicit shutdown() call

    Args:
        args: Command-line arguments (default: None uses sys.argv)
              Can pass custom args for testing or configuration

    Exception handling:
        - KeyboardInterrupt: Allows Ctrl+C to stop the node gracefully
        - All other exceptions: Propagate up (will terminate the program)

    Resource cleanup:
        - destroy_node(): Releases node resources, closes publishers/subscribers
        - shutdown(): Cleans up ROS2 context, frees memory
    """
    # Initialize the ROS2 Python client library
    # This must be called before creating any nodes
    rclpy.init(args=args)

    # Create an instance of the GraphManagementNode
    # This calls __init__(), setting up all subscribers, publishers, and timers
    node = GraphManagementNode()

    try:
        # Enter the ROS2 event loop - blocks here processing callbacks
        # This keeps the node alive and responding to incoming messages
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Allow graceful shutdown on Ctrl+C without printing error traceback
        pass
    finally:
        # Clean up resources regardless of how we exited
        node.destroy_node()  # Release node-specific resources
        rclpy.shutdown()     # Shutdown ROS2 context


# Entry point when script is executed directly (not imported)
if __name__ == '__main__':
    main()
