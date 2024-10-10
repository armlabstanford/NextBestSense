from setuptools import setup
import os

package_name = 'gaussian_splatting_py'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Boshu Lei',
    maintainer_email='sobrmesa121@gmail.com',
    description='Description of your ROS package',
    license='License declaration',
    tests_require=['pytest']
)