#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from scene_graph.msg import GraphObjects, RoomPolygonList, ClassifiedRoom, ClassifiedRooms, RoomWithObjects
import time
import classification_utils
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RoomClassificationNode:
    
    def __init__(self) -> None:
        # self.rooms_sub = rospy.Subscriber('/scene_graph/rooms', RoomPolygonList, self.rooms_callback)
        # self.objects_sub = rospy.Subscriber('scene_graph/graph_objects', GraphObjects, self.objects_callback)
        self.room_with_objects_sub = rospy.Subscriber('/scene/graph/room_with_objects', RoomWithObjects, self.room_with_objects_callback)
        
        self.classified_room_pub = rospy.Publisher('/scene_graph/classified_room', ClassifiedRoom, queue_size=10)
        
        self.rooms = None
        self.configured = False
        
        self.object_occurrences_in_rooms = []
        
        self.rf_classifier = RandomForestClassifier(n_estimators=400, max_depth=10, random_state=42)
        X_train, y_train, self.label_encoder = classification_utils.get_train_dataset()
        self.rf_classifier.fit(X_train, y_train)
        
        print('model fitted')
        self.configured = True
        
        self.mpcat40_mapping = {
            "person": "misc",
            "bicycle": "gym_equipment",
            "car": "misc",
            "motorcycle": "gym_equipment",
            "airplane": "misc",
            "bus": "misc",
            "train": "misc",
            "truck": "misc",
            "boat": "misc",
            "traffic light": "lighting",
            "fire hydrant": "misc",
            "stop sign": "misc",
            "parking meter": "misc",
            "bench": "seating",
            "bird": "misc",
            "cat": "misc",
            "dog": "misc",
            "horse": "misc",
            "sheep": "misc",
            "cow": "misc",
            "elephant": "misc",
            "bear": "misc",
            "zebra": "misc",
            "giraffe": "misc",
            "backpack": "clothes",
            "umbrella": "clothes",
            "handbag": "clothes",
            "tie": "clothes",
            "suitcase": "misc",
            "frisbee": "gym_equipment",
            "skis": "gym_equipment",
            "snowboard": "gym_equipment",
            "sports ball": "gym_equipment",
            "kite": "gym_equipment",
            "baseball bat": "gym_equipment",
            "baseball glove": "gym_equipment",
            "skateboard": "gym_equipment",
            "surfboard": "gym_equipment",
            "tennis racket": "gym_equipment",
            "bottle": "misc",
            "wine glass": "misc",
            "cup": "misc",
            "fork": "misc",
            "knife": "misc",
            "spoon": "misc",
            "bowl": "misc",
            "banana": "misc",
            "apple": "misc",
            "sandwich": "misc",
            "orange": "misc",
            "broccoli": "misc",
            "carrot": "misc",
            "hot dog": "misc",
            "pizza": "misc",
            "donut": "misc",
            "cake": "misc",
            "chair": "chair",
            "couch": "sofa",
            "potted plant": "plant",
            "bed": "bed",
            "dining table": "table",
            "toilet": "toilet",
            "tv": "tv_monitor",
            "laptop": "misc",
            "mouse": "misc",
            "remote": "misc",
            "keyboard": "misc",
            "cell phone": "misc",
            "microwave": "appliances",
            "oven": "appliances",
            "toaster": "appliances",
            "sink": "sink",
            "refrigerator": "appliances",
            "book": "objects",
            "clock": "misc",
            "vase": "misc",
            "scissors": "misc",
            "teddy bear": "misc",
            "hair drier": "misc",
            "toothbrush": "misc"
        }
         
        
    # def rooms_callback(self, msg):
    #     self.rooms = msg
        
    
    def room_with_objects_callback(self, msg):
        
        if not self.configured:
            return
        
        start_time = time.time()
        
        
        detected_objects = []
        for o in msg.objects:
            detected_objects.append(o.data)
            
        print('detected_objects: ', detected_objects)
                    
        classified_room_name = self.classify_room(detected_objects)
        
            
        rospy.loginfo(f'[TIMIMG]: {time.time() - start_time}')
        
            
        classified_room_msg = ClassifiedRoom()
        classified_room_msg.header.stamp = rospy.Time.now()
        classified_room_msg.id = msg.id
        classified_room_msg.label = String(classified_room_name)
        
        self.classified_room_pub.publish(classified_room_msg)
        
    
    
    def classify_room(self, detected_objects):
        
        if len(detected_objects) == 0:
            return
        
        # create normalized vector
        input_vector = [0] * len(classification_utils.get_filtered_mpcat40_list())
        
        for do in detected_objects:
            
            do = self.mpcat40_mapping[do]
            
            i = classification_utils.object_is_in_mpcat40(do)
            
            if i > -1:
                input_vector[i] += 1
        
        score = 0
        for i in input_vector:
            score += i
            
        if score == 0:
            return
        
        for i in range(len(input_vector)):
            input_vector[i] /= score
        
        input_vector = [input_vector]
        input_vector = np.array(input_vector)
        
        predicted_label = self.rf_classifier.predict(input_vector)
        predicted_room = self.label_encoder.inverse_transform(predicted_label)
        
        return predicted_room[0]
    
    
    def get_object_2d_position(self, object):
        center_x = (object.bounding_box[0].x + object.bounding_box[1].x) / 2
        center_y = (object.bounding_box[0].y + object.bounding_box[1].y) / 2
        
        return (center_x, center_y)

    
    def is_point_in_polygon(self, point, polygon):
        """
        Determine if a point is inside a given polygon or not.

        Args:
        point: A tuple (x, y) representing the coordinates of the point to check.
        polygon: A list of tuples [(x1, y1), (x2, y2), ..., (xn, yn)] representing the vertices of the polygon.

        Returns:
        True if the point is inside the polygon, False otherwise.
        """
        x, y = point
        n = len(polygon)
        inside = False

        # p1x, p1y = polygon[0]
        p1x = polygon[0].x
        p1y = polygon[0].y
        for i in range(n + 1):
            # p2x, p2y = polygon[i % n]
            
            p2x = polygon[i % n].x
            p2y = polygon[i % n].y
            
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    
    
if __name__ == '__main__':
    rospy.init_node('room_classification_node')
    
    try:
        RoomClassificationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
