from setuptools import setup
import os
from glob import glob

package_name = 'odom_test'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ========== 核心修改1：安装模型文件到共享目录 ==========
        # (os.path.join('share', package_name, 'model'), glob('odom_test/model/*')),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Odometry test node',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'odom_test_node = odom_test.odom_test:main',
        ],
    },
)