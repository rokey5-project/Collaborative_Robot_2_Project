from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        Node(
            package='albaro',
            executable='manager',
            name='state_manager_node',
            output='screen'
        ),

        # =========================
        # Wakeup Word NodeNode
        # =========================
        Node(
            package='albaro',
            executable='wakeup_word_node',
            name='wakeup_word_node',
            output='screen'
        ),

        # =========================
        # Item Check Node (YOLO)
        # =========================
        Node(
            package='albaro',
            executable='item',
            name='item_check_node',
            output='screen'
        ),

        # =========================
        # Face Age + TTS Node
        # =========================
        Node(
            package='albaro',
            executable='face_age_node',
            name='face_age_node',
            output='screen'
        ),
        # =========================
        # orc Node (depth_check_node)
        # =========================
        Node(
            package='albaro',
            executable='orc',
            name='orc_node',
            output='screen'
        ),
        # =========================
        # pick_place_depth_node
        # =========================
         Node(
            package='albaro',
            executable='picking_depth',
            name='pick_place_depth_node',
            output='screen'
        ),
    ])
