import random
from enum import Enum
import math
import carla
import numpy as np
from gym.spaces import Discrete
from helper.CarlaHelper import post_process_image
import time

# add route planning
import sys
sys.path.append('D:\Code\CARLA_0.9.15\PythonAPI\carla') # tweak to where you put carla
from agents.navigation.global_route_planner import GlobalRoutePlanner


class SensorsTransformEnum(Enum):
    Transform_A = 0  # (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm)
    Transform_B = 1  # (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
    Transform_C = 2  # (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
    Transform_D = 3  # (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm)
    Transform_E = 4  # (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]


class SensorsEnum(Enum):
    CAMERA_RGB = 0
    CAMERA_DEPTH_RAW = 1
    CAMERA_DEPTH_GRAY = 2
    CAMERA_DEPTH_LOG = 3 # Only outline
    CAMERA_SEMANTIC_RAW = 4 # Just RGB
    CAMERA_SEMANTIC_CITYSCAPE = 5 # Real Semantic
    LIDAR = 6 # No image 
    CAMERA_DYNAMIC_VISION = 7 # only reveal dynamic things
    # refer to https://carla.readthedocs.io/en/0.9.15/ref_sensors/#dvs-camera
    CAMERA_DISTORTED = 8 # add noise based on rgb camera


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
    "RENDER": False
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
}
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
    "Weather": carla.WeatherParameters.ClearNoon,
    "DISCRETE_ACTION": True,
    "Debug": False,

    "synchronous": True, # default set synchronous mode to True
    "Autodrive_enabled": False, # TODO to achieve it with Carla Expert for observation
}

# Organized as [throttle, steer, brake, reverse, hand_brake]
DISCRETE_ACTIONS_SMALL = {
    0: [0.0, 0.00, 1.0, False, False],  # Apply Break
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
    14: [0.0, 0.00, 0.0, False, False],  # Coast
}

# DISCRETE_ACTIONS_SMALLER = {
#     0: [0.0, 0.00, 0.0, False, False], # Coast
#     1: [0.0, -0.15, 0.0, False, False], # Turn Left
#     2: [0.0, 0.15, 0.0, False, False], # Turn Right
#     3: [0.2, 0.00, 0.0, False, False], # Accelerate
#     4: [-0.3, 0.00, 0.0, False, False], # Decelerate
#     5: [0.0, 0.00, 1.0, False, False], # Brake
#     6: [0.2, 0.15, 0.0, False, False], # Turn Right + Accelerate
#     7: [0.2, -0.15, 0.0, False, False], # Turn Left + Accelerate
#     8: [-0.3, 0.10, 0.0, False, False], # Turn Right + Decelerate
#     9: [-0.3, -0.15, 0.0, False, False], # Turn Left + Decelerate
# }

DISCRETE_ACTIONS_SMALLER = {
    0: [0.0, 0.00, 0.0, False, False], # Coast
    1: [0.0, -0.1, 0.0, False, False], # Turn Left
    2: [0.0, 0.1, 0.0, False, False], # Turn Right
    3: [1.0, 0.00, 0.0, False, False], # Accelerate
    4: [0.0, 0.00, 1.0, False, False], # Brake
}

DISCRETE_ACTIONS = DISCRETE_ACTIONS_SMALL

class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()

class BaseExperiment:
    def __init__(self, config=BASE_EXPERIMENT_CONFIG):
        self.experiment_config = config
        self.observation = {}
        self.observation["camera"] = []
        self.observation_space = None
        self.action = None
        self.action_space = None

        self.hero = None
        self.spectator = None
        self.spawn_point_list = []
        self.vehicle_list = []
        self.start_location = None
        # self.end_location = None
        self.current_w = None
        self.hero_model = ''.join(self.experiment_config["hero_vehicle_model"])
        self.set_observation_space()
        self.set_action_space()
        self.max_idle = 40 # seconds
        self.max_ep_time = 120 # seconds
        self.timer_idle = CustomTimer()
        self.timer_ep = CustomTimer()
        self.t_idle_start = None
        self.t_ep_start = None

        self.route = None
        self.speed_limit = 30 # kmh


    def get_experiment_config(self):

        return self.experiment_config

    def set_observation_space(self):

        """
        observation_space_option: Camera Image
        :return: observation space:
        """
        raise NotImplementedError

    def get_observation_space(self):

        """
        :return: observation space
        """
        return self.observation_space

    def set_action_space(self):

        """
        :return: None. In this experiment, it is a discrete space
        """
        self.action_space = Discrete(len(DISCRETE_ACTIONS))

    def get_action_space(self):

        """
        :return: action_space. In this experiment, it is a discrete space
        """
        return self.action_space

    def set_server_view(self,core):

        """
        Set server view to be behind the hero
        :param core:Carla Core
        :return:
        """
        # spectator following the car
        transforms = self.hero.get_transform()
        server_view_x = self.hero.get_location().x - 5 * transforms.get_forward_vector().x
        server_view_y = self.hero.get_location().y - 5 * transforms.get_forward_vector().y
        server_view_z = self.hero.get_location().z + 3
        server_view_pitch = transforms.rotation.pitch
        server_view_yaw = transforms.rotation.yaw
        server_view_roll = transforms.rotation.roll
        self.spectator = core.get_core_world().get_spectator()
        self.spectator.set_transform(
            carla.Transform(
                carla.Location(x=server_view_x, y=server_view_y, z=server_view_z),
                carla.Rotation(pitch=server_view_pitch,yaw=server_view_yaw,roll=server_view_roll),
            )
        )

    def get_speed(self):
        """
        Compute speed of a vehicle in Km/h.

            :param vehicle: the vehicle for which speed is calculated
            :return: speed as a float in Km/h
        """
        vel = self.hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def get_done_status(self):
        done_collision = self.observation["collision"]
        # done = self.observation["collision"] or not self.check_lane_type(map)
        
        # Done when keep low speed for a long time
        if self.get_speed() < 4.0:
            idle_time = self.timer_idle.time()
            done_idle = (idle_time - self.t_idle_start) > self.max_idle
        else:
            done_idle = False
            self.t_idle_start = self.timer_idle.time()
        
        # Done when total time is too long
        ep_time = self.timer_ep.time()
        done_max_time = (ep_time - self.t_ep_start) > self.max_ep_time

        # Done when fall out the world
        done_falling = self.hero.get_location().z < -0.5 
        
        return done_collision or done_idle or done_max_time or done_falling

    def process_observation(self, core, observation):

        """
        Main function to do all the post processing of observations. This is an example.
        :param core:
        :param observation:
        :return:
        """
        observation['camera'] = post_process_image(
                                            observation['camera'],
                                            normalized = self.experiment_config["SENSOR_CONFIG"]["CAMERA_NORMALIZED"][0],
                                            grayscale = self.experiment_config["SENSOR_CONFIG"]["CAMERA_GRAYSCALE"][0]
            )

        return observation

    def get_observation(self, core):

        info = {}
        obs = self.observation

        if len(self.experiment_config["SENSOR_CONFIG"]["SENSOR"]) != len(self.experiment_config["OBSERVATION_CONFIG"]["CAMERA_OBSERVATION"]):
            raise Exception("You need to specify the CAMERA_OBSERVATION for each sensor.")
        
        for i in range(0,len(self.experiment_config["SENSOR_CONFIG"]["SENSOR"])):
            if self.experiment_config["OBSERVATION_CONFIG"]["CAMERA_OBSERVATION"][i]:
                # np.concatenate(obs['camera'], core.get_camera_data())
                obs['camera'].append(core.get_camera_data())

        # obs['camera'] = core.get_camera_data()
                
        if self.experiment_config["OBSERVATION_CONFIG"]["COLLISION_OBSERVATION"]:
            obs["collision"] = core.get_collision_data()
        if self.experiment_config["OBSERVATION_CONFIG"]["LOCATION_OBSERVATION"]:
            loc = self.hero.get_transform().location
            obs["location"] = [loc.x, loc.y, loc.z]
        if self.experiment_config["OBSERVATION_CONFIG"]["LANE_OBSERVATION"]:
            obs["lane_invasion"] = core.get_lane_data()
        if self.experiment_config["OBSERVATION_CONFIG"]["GNSS_OBSERVATION"]:
            obs["gnss"] = core.get_gnss_data()
        if self.experiment_config["OBSERVATION_CONFIG"]["IMU_OBSERVATION"]:
            obs["imu"] = core.get_imu_data()
        if self.experiment_config["OBSERVATION_CONFIG"]["RADAR_OBSERVATION"]:
            obs["radar"] = core.get_radar_data()
        if self.experiment_config["OBSERVATION_CONFIG"]["BIRDVIEW_OBSERVATION"]:
            obs["birdview"] = core.get_birdview_data()

        info["control"] = {
            "steer": self.action.steer,
            "throttle": self.action.throttle,
            "brake": self.action.brake,
            "reverse": self.action.reverse,
            "hand_brake": self.action.hand_brake,
        }

        # Add observation about navigation waypoint
        angle, distance = None, None
        while angle is None:
            try:
                # connect
                angle, distance = self.get_closest_wp_forward()
            except:
                pass

        obs["waypoint_angle"] = angle
        obs["waypoint_distance"] = distance
        self.past_distance_to_route = distance

        return obs, info

    def update_measurements(self, core):

        if len(self.experiment_config["SENSOR_CONFIG"]["SENSOR"]) != len(self.experiment_config["OBSERVATION_CONFIG"]["CAMERA_OBSERVATION"]):
                raise Exception("You need to specify the CAMERA_OBSERVATION for each sensor.")
        
        for i in range(0,len(self.experiment_config["SENSOR_CONFIG"]["SENSOR"])):
            if self.experiment_config["OBSERVATION_CONFIG"]["CAMERA_OBSERVATION"][i]:
                core.update_camera()
        if self.experiment_config["OBSERVATION_CONFIG"]["COLLISION_OBSERVATION"]:
            core.update_collision()
        if self.experiment_config["OBSERVATION_CONFIG"]["LANE_OBSERVATION"]:
            core.update_lane_invasion()


    def update_actions(self, action, hero):
        if action is None:
            self.action = carla.VehicleControl() # default：do nothing 
        else:
            action = DISCRETE_ACTIONS[int(action)]

            # self.action.throttle = float(np.clip(self.past_action.throttle + action[0], 0, 0.5))
            # self.action.steer = float(np.clip(self.past_action.steer + action[1], -0.7, 0.7))

            self.action.throttle = float(np.clip(action[0], 0, 0.5))
            self.action.steer = float(np.clip(action[1], -0.7, 0.7))
            self.action.brake = float(np.clip(action[2], 0, 1))
            
            # Throttle when brake result in nothing
            # if self.action.brake:
            #     self.action.throttle = float(0)
            # else:
            #     self.action.throttle = float(np.clip(self.past_action.throttle + action[0], 0, 0.5))

            self.action.reverse = action[3]
            self.action.hand_brake = action[4]
            self.past_action = self.action
        
        hero.apply_control(self.action)

    def compute_reward(self, core, observation):

        """
        :param core:
        :param observation:
        :return:
        """

        print("This is a base experiment. Make sure you make you own reward computing function")
        return NotImplementedError

    def initialize_reward(self, core):

        """
        Generic initialization of reward function
        :param core:
        :return:
        """
        print("This is a base experiment. Make sure you make you own reward initialization function")
        raise NotImplementedError


    # ==============================================================================
    # -- Hero -----------------------------------------------------------
    # ==============================================================================
    def spawn_hero(self, world, transform, autopilot=False):

        """
        This function spawns the hero vehicle. It makes sure that if a hero exists, it destroys the hero and respawn
        :param core:
        :param transform: Hero start location
        :param autopilot: Autopilot Status
        :return:
        """

        self.spawn_points = world.get_map().get_spawn_points()

        self.hero_blueprints = world.get_blueprint_library().find(self.hero_model)
        self.hero_blueprints.set_attribute("role_name", "hero")

        # self.end_location = self.spawn_points[self.experiment_config["end_pos_spawn_id"]]

        if self.hero is not None:
            self.hero.destroy()
            self.hero = None

        i = 0
        random.shuffle(self.spawn_points, random.random)
        # Keep finding spawn point until hero successfully spawned
        while True:
            next_spawn_point = self.spawn_points[i % len(self.spawn_points)]
            self.hero = world.try_spawn_actor(self.hero_blueprints, next_spawn_point)
            if self.hero is not None:
                break
            else:
                print("Could not spawn Hero, changing spawn point")
                i+=1

        world.tick()
        print("Hero spawned!")

        # initialize params
        self.start_location = self.spawn_points[i].location
        self.route = self.select_random_route(world)
        self.past_action = carla.VehicleControl(0.0, 0.00, 0.0, False, False)
        self.t_idle_start = self.timer_idle.time()
        self.t_ep_start = self.timer_ep.time()

    def get_hero(self):

        """
        Get hero vehicle
        :return:
        """
        return self.hero

    # ==============================================================================
    # -- Tick -----------------------------------------------------------
    # ==============================================================================

    def experiment_tick(self, core, world, action):

        """
        This is the "tick" logic.
        :param core:
        :param action:
        :return:
        """

        world.tick()
        self.update_measurements(core)
        self.update_actions(action, self.hero)

    
    def select_random_route(self, world):
        '''
        retruns a random route for the car/veh
        out of the list of possible locations locs
        where distance is longer than 100 waypoints
        '''  
        hero = self.hero  
        point_a = hero.get_transform().location #we start at where the car is or last waypoint
        sampling_resolution = 1
        grp = GlobalRoutePlanner(world.get_map(), sampling_resolution)
        # now let' pick the longest possible route
        min_distance = 100
        result_route = None
        route_list = []
        for loc in world.get_map().get_spawn_points(): # we start trying all spawn points 
                                                            #but we just exclude first at zero index
            cur_route = grp.trace_route(point_a, loc.location)
            if len(cur_route) > min_distance:
                route_list.append(cur_route)
        result_route = random.choice(route_list)
        return result_route
    
    def get_closest_wp_forward(self):
        '''
        this function is to find the closest point looking forward
        if there in no points behind, then we get first available
        '''

        # first we create a list of angles and distances to each waypoint
        # yeah - maybe a bit wastefull
        hero = self.hero
        points_ahead = []
        points_behind = []
        for i, wp in enumerate(self.route):
            #get angle
            vehicle_transform = hero.get_transform()
            wp_transform = wp[0].transform
            distance = ((wp_transform.location.y - vehicle_transform.location.y)**2 + (wp_transform.location.x - vehicle_transform.location.x)**2)**0.5
            angle = math.degrees(math.atan2(wp_transform.location.y - vehicle_transform.location.y,
                                wp_transform.location.x - vehicle_transform.location.x)) -  vehicle_transform.rotation.yaw
            if angle>360:
                angle = angle - 360
            elif angle <-360:
                angle = angle + 360

            if angle>180:
                angle = -360 + angle
            elif angle <-180:
                angle = 360 - angle 
            if abs(angle)<=90:
                points_ahead.append([i,distance,angle])
            else:
                points_behind.append([i,distance,angle])
        # now we pick a point we need to get angle to 
        if len(points_ahead)== 0:
            closest = min(points_behind, key=lambda x: x[1])
            if closest[2]>0:
                closest = [closest[0],closest[1],90]
            else:
                closest = [closest[0],closest[1],-90] 
        else:
            closest = min(points_ahead, key=lambda x: x[1])
            # move forward if too close
            for i, point in enumerate(points_ahead):
                if point[1]>=10 and point[1]<20:
                    closest = point
                    break
            return closest[2]/90.0, closest[1] # we convert angle to [-1 to +1] and also return distance
