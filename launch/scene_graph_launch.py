from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """Launch file for scene graph room classification nodes."""
    
    # Image Segmentation Node
    image_segmentation_node = Node(
        package='scene_graph_room_classification',
        executable='image_segmentation_node',
        name='image_segmentation_node',
        output='screen',
        parameters=[],
        remappings=[]
    )
    
    # Object Localization Node
    object_localization_node = Node(
        package='scene_graph_room_classification',
        executable='object_localization_node',
        name='object_localization_node',
        output='screen',
        parameters=[],
        remappings=[]
    )
    
    # Scene Graph Fusion Node
    scene_graph_fusion_node = Node(
        package='scene_graph_room_classification',
        executable='scene_graph_fusion_node',
        name='scene_graph_fusion_node',
        output='screen',
        parameters=[],
        remappings=[]
    )
    
    return LaunchDescription([
        image_segmentation_node,
        object_localization_node,
        scene_graph_fusion_node,
    ])
