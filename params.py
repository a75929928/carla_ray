"""
    store 
"""
from enum import Enum
import carla

class SensorsTransformEnum(Enum):
    Transform_A = 0  # (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm)
    Transform_B = 1  # (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
    Transform_c = 2  # (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
    Transform_D = 3  # (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm)
    Transform_E = 4  # (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]


class SensorsEnum(Enum):
    CAMERA_RGB = 0
    CAMERA_DEPTH_RAW = 1
    CAMERA_DEPTH_GRAY = 2
    CAMERA_DEPTH_LOG = 3
    CAMERA_SEMANTIC_RAW = 4
    CAMERA_SEMANTIC_CITYSCAPE = 5
    LIDAR = 6
    CAMERA_DYNAMIC_VISION = 7
    CAMERA_DISTORTED = 8


BASE_SERVER_VIEW_CONFIG = {
    "server_view_x_offset": 00,
    "server_view_y_offset": 00,
    "server_view_height": 200,
    "server_view_pitch": -90,
}

BASE_SENSOR_CONFIG = {
    "SENSOR": [SensorsEnum.CAMERA_DEPTH_RAW],
    "SENSOR_TRANSFORM": [SensorsTransformEnum.Transform_A],
    "CAMERA_X": 84,
    "CAMERA_Y": 84,
    "CAMERA_FOV": 60,
    "CAMERA_NORMALIZED": [True],
    "CAMERA_GRAYSCALE": [True],
    "FRAMESTACK": 1,
}

BASE_BIRDVIEW_CONFIG = {
    "SIZE": 300,
    "RADIUS": 20,
    "FRAMESTACK": 4,
    # "RENDER": False
}

BASE_OBSERVATION_CONFIG = {
    "CAMERA_OBSERVATION": [False],
    "COLLISION_OBSERVATION": True,
    "LOCATION_OBSERVATION": True,
    "RADAR_OBSERVATION": False,
    "IMU_OBSERVATION": False,
    "LANE_OBSERVATION": True,
    "GNSS_OBSERVATION": False,
    "BIRDVIEW_OBSERVATION": False,
    "COMMUNICATION_OBSERVATION": False
}

"""
Avaliable Weather:
    ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon, SoftRainNoon, 
    MidRainyNoon, HardRainNoon, ClearSunset, CloudySunset, 
    WetSunset, WetCloudySunset, SoftRainSunset, MidRainSunset, HardRainSunset. 
"""
BASE_EXPERIMENT_CONFIG = {
    "OBSERVATION_CONFIG": BASE_OBSERVATION_CONFIG,
    "Server_View": BASE_SERVER_VIEW_CONFIG,
    "SENSOR_CONFIG": BASE_SENSOR_CONFIG,
    "BIRDVIEW_CONFIG": BASE_BIRDVIEW_CONFIG,
    "server_map": "Town02_Opt",
    "quality_level": "Low",  # options are low or Epic #ToDO. This does not do anything + change to enum
    "Disable_Rendering_Mode": False,  # If you disable, you will not get camera images
    "n_vehicles": 0,
    "n_walkers": 0,
    "end_pos_spawn_id": 45,  # 34,
    "hero_vehicle_model": "vehicle.lincoln.mkz_2017",
    
    "Weather": "ClearNoon",

    "DISCRETE_ACTION": True,
    # "Debug": False,

    "synchronous": True, # default set synchronous mode to True
    # All heroes would be controlled by traffic agent
    # TODO Take (obs, action) of background agents into consideration
    "Autodrive_enabled": False, 

    "n_heroes": 1,
    "allow_respawn": False,
}

# Organized as [throttle, steer, brake, reverse, hand_brake]
DISCRETE_ACTIONS_SMALL = {
    0: [0.0, 0.00, 0.0, False, False],  # Coast
    1: [1.0, 0.00, 0.0, False, False],  # Straight
    2: [1.0, -0.70, 0.0, False, False],  # Right + Accelerate
    3: [1.0, -0.50, 0.0, False, False],  # Right + Accelerate
    4: [1.0, -0.30, 0.0, False, False],  # Right + Accelerate
    5: [1.0, -0.10, 0.0, False, False],  # Right + Accelerate
    6: [1.0, 0.10, 0.0, False, False],  # Left+Accelerate
    7: [1.0, 0.30, 0.0, False, False],  # Left+Accelerate
    8: [1.0, 0.50, 0.0, False, False],  # Left+Accelerate
    9: [1.0, 0.70, 0.0, False, False],  # Left+Accelerate
    10: [0.0, -0.70, 1.0, False, False],  # Left+Stop
    11: [0.0, -0.23, 1.0, False, False],  # Left+Stop
    12: [0.0, 0.23, 1.0, False, False],  # Right+Stop
    13: [0.0, 0.70, 1.0, False, False],  # Right+Stop
    14: [0.0, 0.00, 1.0, False, False],  # Brake
}

DISCRETE_ACTIONS_SMALLER = {
    0: [0.0, 0.00, 0.0, False, False], # Coast
    1: [0.0, -0.1, 0.0, False, False], # Turn Left
    2: [0.0, 0.1, 0.0, False, False], # Turn Right
    3: [1.0, 0.00, 0.0, False, False], # Accelerate
    4: [0.0, 0.00, 1.0, False, False], # Brake
}

DISCRETE_ACTIONS = DISCRETE_ACTIONS_SMALL