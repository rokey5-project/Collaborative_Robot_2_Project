from setuptools import find_packages, setup

package_name = 'albaro'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='rokey@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'face_age_node = albaro.face_age_node:main',
            'picking_node = albaro.picking_node:main',
            'wakeup_word_node = albaro.wakeup_word_node:main',
            'detect_shelves_node = albaro.detect_shelves_node:main',
            'picking_depth = albaro.pick_place_depth:main',
            'item =albaro.item_check_node:main',
            'manager = albaro.state_manager_node:main',
            'orc = albaro.orc_node:main',
        ],
    },
)
