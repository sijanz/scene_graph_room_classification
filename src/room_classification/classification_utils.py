import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.feature_selection import RFE
import numpy as np

folder_path = '/home/nes/catkin_ws/src/scene_graph/src/room_classification/datasets/matterport/v1/scans'

def get_filtered_mpcat40_list():
    data = """
            mpcat40index	mpcat40	hex	wnsynsetkey	nyu40	skip	labels
            0	void	#ffffff		void		remove,void
            1	wall	#aec7e8	wall.n.01,baseboard.n.01,paneling.n.01	wall		wallpaper
            2	floor	#708090	floor.n.01,rug.n.01,mat.n.01,bath_mat.n.01,landing.n.01	floor,floor mat		
            3	chair	#98df8a	chair.n.01,beanbag.n.01	chair		
            4	door	#c5b0d5	door.n.01,doorframe.n.01,doorway.n.01,doorknob.n.01,archway.n.01	door		garage door
            5	table	#ff7f0e	table.n.02,dressing.n.04	table,desk	counter.n.01	
            6	picture	#d62728	picture.n.01,photograph.n.01,picture_frame.n.01	picture		
            7	cabinet	#1f77b4	cabinet.n.01,cupboard.n.01	cabinet		
            8	cushion	#bcbd22	cushion.n.03	pillow		couch cushion
            9	window	#ff9896	window.n.01,windowsill.n.01,window_frame.n.01,windowpane.n.02,window_screen.n.01	window		
            10	sofa	#2ca02c	sofa.n.01	sofa		
            11	bed	#e377c2	bed.n.01,bedpost.n.01,bedstead.n.01,headboard.n.01,footboard.n.01,bedspread.n.01,mattress.n.01,sheet.n.03	bed		
            12	curtain	#de9ed6	curtain.n.01	curtain,shower curtain		curtain rod,shower curtain rod,shower rod
            13	chest_of_drawers	#9467bd	chest_of_drawers.n.01,drawer.n.01	dresser,night stand		
            14	plant	#8ca252	plant.n.02			
            15	sink	#843c39	sink.n.01	sink		
            16	stairs	#9edae5	step.n.04,stairway.n.01,stairwell.n.01			
            17	ceiling	#9c9ede	ceiling.n.01,roof.n.01	ceiling		
            18	toilet	#e7969c	toilet.n.01,bidet.n.01	toilet		
            19	stool	#637939	stool.n.01			
            20	towel	#8c564b	towel.n.01	towel		
            21	mirror	#dbdb8d	mirror.n.01	mirror		
            22	tv_monitor	#d6616b	display.n.06	television		
            23	shower	#cedb9c	shower.n.01,showerhead.n.01			
            24	column	#e7ba52	column.n.07,post.n.04			
            25	bathtub	#393b79	bathtub.n.01	bathtub		
            26	counter	#a55194	countertop.n.01,counter.n.01,kitchen_island.n.01	counter		
            27	fireplace	#ad494a	fireplace.n.01,mantel.n.01			
            28	lighting	#b5cf6b	lamp.n.02,lampshade.n.01,light.n.02,chandelier.n.01,spotlight.n.02	lamp		
            29	beam	#5254a3	beam.n.02			
            30	railing	#bd9e39	railing.n.01,bannister.n.02			
            31	shelving	#c49c94	bookshelf.n.01,shelf.n.01,rack.n.05	shelves		
            32	blinds	#f7b6d2	window_blind.n.01	blinds		
            33	gym_equipment	#6b6ecf	sports_equipment.n.01,treadmill.n.01,exercise_bike.n.01			
            34	seating	#ffbb78	bench.n.01,seat.n.03			
            35	board_panel	#c7c7c7	panel.n.01	whiteboard		board
            36	furniture	#8c6d31	furniture.n.01	otherfurniture		
            37	appliances	#e7cb94	home_appliance.n.01,stove.n.02,dryer.n.01	refridgerator		washing machine and dryer
            38	clothes	#ce6dbd	clothing.n.01	clothes		
            39	objects	#17becf	physical_object.n.01,material.n.01	books,paper,box,bag,otherprop	structure.n.01,way.n.06,vent.n.01,unknown.n.01,pool.n.01	
            40	misc	#7f7f7f		person,otherstructure	unknown.n.01	
            41	unlabeled	#000000	unknown.n.01			unknown
            """
            
    lines = data.strip().split('\n')
    return [line.split('\t')[1] for line in lines[1:]]
        
room_classes = [
    ['a', 'bathroom'], # (should have a toilet and a sink)
    ['b', 'bedroom'],
    ['c', 'closet'],
    ['d', 'dining room'], # (includes “breakfast rooms” other rooms people mainly eat in)
    ['e', 'entryway/foyer/lobby'], # (should be the front door, not any door)
    ['f', 'familyroom'], # (should be a room that a family hangs out in, not any area with couches)
    ['g', 'garage'],
    ['h', 'hallway'],
    ['i', 'library'], # (should be room like a library at a university, not an individual study)
    ['j', 'laundryroom/mudroom'], # (place where people do laundry, etc.)
    ['k', 'kitchen'],
    ['l', 'living room'], # (should be the main “showcase” living room in a house, not any area with couches)
    ['m', 'meetingroom/conferenceroom'],
    ['n', 'lounge'], # (any area where people relax in comfy chairs/couches that is not the family room or living room
    ['o', 'office'], # (usually for an individual, or a small set of people)
    ['p', 'porch/terrace/deck/driveway'], # (must be outdoors on ground level)
    ['r', 'rec/game'], # (should have recreational objects, like pool table, etc.)
    ['s', 'stairs'],
    ['t', 'toilet'], # (should be a small room with ONLY a toilet)
    ['u', 'utilityroom/toolroom'], 
    ['v', 'tv'], # (must have theater-style seating)
    ['w', 'workout/gym/exercise'],
    ['x', 'outdoor areas'], # containing grass, plants, bushes, trees, etc.
    ['y', 'balcony'], # (must be outside and must not be on ground floor)
    ['z', 'other room'], # (it is clearly a room, but the function is not clear)
    ['B', 'bar'],
    ['C', 'classroom'],
    ['D', 'dining booth'],
    ['S', 'spa/sauna'],
    ['Z', 'junk'], # (reflections of mirrors, random points floating in space, etc.)
    ['-', 'no label']
]

room_label_blacklist = ['utilityroom/toolroom', 'tv', 'toilet', 'stairs', 'spa/sauna', 'rec/game', 'outdoor areas', 'other room', 'library', 'junk', 'bar', 'garage', 'classroom',
                        'entryway/foyer/lobby', 'familyroom', 'balcony', 'laundryroom/mudroom', 'meetingroom/conferenceroom', 'workout/gym/exercise', 'dining booth', 'porch/terrace/deck/driveway', 'lounge']
object_label_blacklist = ['void', 'wall', 'floor', 'objects', 'misc', 'unlabeled']

room_label_blacklist_ = [
    'closet',
    'dining room', # (includes “breakfast rooms” other rooms people mainly eat in)
    'entryway/foyer/lobby', # (should be the front door, not any door)
    'familyroom', # (should be a room that a family hangs out in, not any area with couches)
    'garage',
    'library', # (should be room like a library at a university, not an individual study)
    'laundryroom/mudroom', # (place where people do laundry, etc.)
    'kitchen',
    'living room', # (should be the main “showcase” living room in a house, not any area with couches)
    'meetingroom/conferenceroom',
    'lounge', # (any area where people relax in comfy chairs/couches that is not the family room or living room
    'office', # (usually for an individual, or a small set of people)
    'porch/terrace/deck/driveway', # (must be outdoors on ground level)
    'rec/game', # (should have recreational objects, like pool table, etc.)
    'stairs',
    'toilet', # (should be a small room with ONLY a toilet)
    'utilityroom/toolroom', 
    'tv', # (must have theater-style seating)
    'workout/gym/exercise',
    'outdoor areas', # containing grass, plants, bushes, trees, etc.
    'balcony', # (must be outside and must not be on ground floor)
    'other room', # (it is clearly a room, but the function is not clear)
    'bar',
    'classroom',
    'dining booth',
    'spa/sauna',
    'junk', # (reflections of mirrors, random points floating in space, etc.)
    'no label'
]

mpcat40_mapping = [
    ['tv', 'tv_monitor'],
    ['potted plant', 'plant'],
    ['couch', 'sofa'],
    ['bottle'],
    ['mouse'],
    ['laptop'],
    ['cup'],
    ['book'],
    ['remote'],
    ['handbag', 'clothes'],
    ['dining table', 'table'],
    ['keyboard'],
    ['scissors'], 
    ['refrigerator', 'appliances']
]

def get_train_dataset():
    filtered_mpcat40_list = get_filtered_mpcat40_list()
    
    all_rooms = []
    all_objects = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.house'):
                
                # Construct full file path
                file_path = os.path.join(root, file)
                # print(file_path)
                
                categories = []
                
                # rooms = []
                # objects = []
                
                # [[room_index, room_label, [object_class, ...]], ...]
                rooms_with_objects = []
                
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip().split()
                        
                        if line[0] == 'C':
                            categories.append(line[5])
                            
                    f.seek(0)
                    
                    for line in f:
                        line = line.strip().split()
                        
                        if line[0] == 'R':
                            label = line[5]
                            verbose_label = label
                            
                            for l in room_classes:
                                if label == l[0]:
                                    verbose_label = l[1]
                                    
                            rooms_with_objects.append([line[1], verbose_label, []])
                            
                    f.seek(0)
                
                
                    for line in f:
                        line = line.strip().split()
                        
                        # if line[0] == 'R':
                        #     room_label = line[5]
                            
                        #     x_min = float(line[9])
                        #     y_min = float(line[10])
                        #     x_max = float(line[12])
                        #     y_max = float(line[13])
                                
                        if line[0] == 'O':
                            
                            region_index = int(line[2])
                            category_index = int(line[3])
                            
                            # get label
                            label = categories[category_index]
                            
                            if label in object_label_blacklist:
                                continue
                            
                            # find corresponding room
                            # print('size of room_with_objects: ', len(rooms_with_objects))
                            for room in rooms_with_objects:
                                if region_index == int(room[0]):
                                    room[2].append(label)
                                    
                for room in rooms_with_objects:
                    # print(room)
                    
                    feature_vector = [0] * len(filtered_mpcat40_list)
                    
                    for i, l in enumerate(filtered_mpcat40_list):
                        for o in room[2]:
                            # print(o, l)
                            
                            if o == l:
                                feature_vector[i] += 1
                                
                    # print(feature_vector)
                            
                    count = 0
                    for c in feature_vector:
                        count += c
                        
                    omit = False
                    for i in range(len(feature_vector)):
                        if count > 0:
                            feature_vector[i] /= count
                        else:
                            omit = True
                            
                    if room[1] in room_label_blacklist:
                        omit = True
                    
                        
                    if not omit:
                        all_objects.append(feature_vector)
                        all_rooms.append(room[1])
                        
                        # print(feature_vector)

    # Step 1: Dictionary to track occurrences of each string
    occurrences = {}

    # Step 2: List to store the indices to keep
    indices_to_keep = []

    # Step 3: Iterate through the strings list
    for i, s in enumerate(all_rooms):
        if occurrences.get(s, 0) < 70:
            indices_to_keep.append(i)
            occurrences[s] = occurrences.get(s, 0) + 1

    # Step 4: Filter both lists based on the indices_to_keep
    all_rooms = [all_rooms[i] for i in indices_to_keep]
    all_objects = [all_objects[i] for i in indices_to_keep]

    room_occurences = []
    for room in all_rooms:
        
        in_list = -1
        
        for i, o in enumerate(room_occurences):
            if o[0] == room:
                in_list = i
                break
        
        if in_list != -1:
            room_occurences[in_list][1] += 1
        else:
            room_occurences.append([room, 1])
            
    room_occurences = sorted(room_occurences, key=lambda x: x[1], reverse=True)
    # for room in room_occurences:
    #     print(room)
    # print(len(all_rooms))
    # print(len(all_objects))

    # exit()
                

    room_objects = np.array(all_objects)

    # # Corresponding room labels
    # room_labels = ['living room', 'bedroom', 'kitchen', 'bathroom', 'office']

    room_labels = all_rooms

    # Encode labels to numerical format
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(room_labels)

    # # Display the encoded labels
    # print("Encoded Labels:")
    # print(y)


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(room_objects, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, y_train, label_encoder


def object_is_in_mpcat40(detected_object):
    
    for i in mpcat40_mapping:
        if detected_object == i[0]:
            if len(i) > 1:
                detected_object = i[1]
    
    mpcat40_list = get_filtered_mpcat40_list()
    
    for i in range(len(mpcat40_list)):
        if mpcat40_list[i] == detected_object:
            return i
    
    return -1