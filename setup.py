from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'scene_graph_room_classification'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'image_segmentation_node = scene_graph_room_classification.image_segmentation.image_segmentation_node:main',
            'object_localization_node = scene_graph_room_classification.object_localization.object_localization_node:main',
            'scene_graph_fusion_node = scene_graph_room_classification.scene_graph_fusion.scene_graph_fusion_node:main',
        ],
    },
)
