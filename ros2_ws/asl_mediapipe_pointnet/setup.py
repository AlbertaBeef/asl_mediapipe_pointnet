from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'asl_mediapipe_pointnet'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files.
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='AlbertaBeef',
    maintainer_email='grouby177@gmail.com',
    description='ASL recognition using mediapipe and pointnet.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'asl_controller_twist_node = asl_mediapipe_pointnet.asl_controller_twist_node:main', 
            'asl_controller_joints_node = asl_mediapipe_pointnet.asl_controller_joints_node:main', 
            'asl_controller_pose_node = asl_mediapipe_pointnet.asl_controller_pose_node:main', 
        ],
    },
)
