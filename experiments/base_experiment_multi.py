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

'''
    self.hero -> all heroes in {id: <carla.hero>} 
    _hero in function -> certain hero <carla.hero>
'''
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
    # All heroes would be controlled by traffic agent
    # TODO Take (obs, action) of background agents into consideration
    "Autodrive_enabled": False, 

    "n_heroes": 1
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

class CustomTimer:
    def __init__(self):
        self.start_time = time()

    def past_time(self):
        return time() - self.start_time

class BaseExperiment:
    def __init__(self, config=BASE_EXPERIMENT_CONFIG):
        self.experiment_config = config
        
        # ---- TODO need to be modified ---- 
        # self.observation[hero_id]["camera"] = [] # dict->dict how to initialize every dict inside?

        # Prameters
        self.max_idle = 40 # seconds
        self.max_ep_time = 120 # seconds
        self.speed_limit = 30 # kmh
        self.observation_space = None
        self.action_space = None
        self.hero_model = ''.join(self.experiment_config["hero_vehicle_model"]) # TODO differ hero models

        # ---- Unique variable ---- 
        self.spectator = None # spectator randomly choose hero

        # ---- Multi-agent part ----
        self.hero = {}
        self.route = {}
        self.observation = {}
        self.action = {}
        
        self.spawn_point_list = []
        self.vehicle_list = []
        self.start_location = {}
        
        self.t_idle_start = {}
        self.t_ep_start = {}

        self.current_w = {}

        # initialize
        self.set_observation_space()
        self.set_action_space()
        
        
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

    def set_server_view(self, core):

        """
        Set server view to be behind the hero
        :param core:Carla Core
        :return:
        """
        # randomly select a hero
        hero_id = random.choice(list(self.hero.keys()))
        # spectator following the car
        hero = self.hero[hero_id]
        transforms = hero.get_transform()
        server_view_x = hero.get_location().x - 5 * transforms.get_forward_vector().x
        server_view_y = hero.get_location().y - 5 * transforms.get_forward_vector().y
        server_view_z = hero.get_location().z + 3
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

    def get_speed(self, hero_id):
        """
        Compute speed of a vehicle in Km/h.

            :param vehicle: the vehicle for which speed is calculated
            :return: speed as a float in Km/h
        """
        hero = self.hero[hero_id]
        vel = hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def get_done_status(self):
        done = {}
        for hero_id in self.hero:
            _done = self.get_done_status_single(hero_id)
            done.update({hero_id: _done})
        return done
    
    def get_done_status_single(self, hero_id):
        # ---- Done when vehicle crashes something ---- 
        done_collision = self.observation[hero_id]["collision"]
        # done = self.observation[hero_id]["collision"] or not self.check_lane_type(map)
        
        # ---- Done when hero keeps low speed for a long time ---- 
        # Every tick cost 0.1s, so t_idle_start could reflect the start time of hero being idle 
        if self.get_speed(hero_id) < 4.0:
            done_idle = (time.time() - self.t_idle_start[hero_id]) > self.max_idle
        else:
            done_idle = False
            self.t_idle_start[hero_id] = time.time()
        
        # ---- Done when vehicle lives too long ---- 
        done_max_time = (time.time() - self.t_ep_start[hero_id]) > self.max_ep_time

        # ---- Done when vehicle falls out the world ---- 
        hero = self.hero[hero_id]
        done_falling = hero.get_location().z < -0.5 
        
        return done_collision or done_idle or done_max_time or done_falling

    def process_observation(self, core, observation):

        """
        Main function to do all the post processing of observations. This is an example.
        :param core:
        :param observation:
        :return:
        """
        sensor_config = self.experiment_config["SENSOR_CONFIG"]
        observation['camera'] = post_process_image(
                                            observation['camera'],
                                            normalized = sensor_config["CAMERA_NORMALIZED"][0],
                                            grayscale = sensor_config["CAMERA_GRAYSCALE"][0]
            )

        return observation

    def get_observation(self, core):
        observation = {}
        info = {}
        for hero_id in self.hero:
            _observation, _info = self.get_observation_single(core, hero_id)
            observation.update({hero_id: _observation})
            info.update({hero_id: _info})
        return observation, info
    
    def get_observation_single(self, core, hero_id):

        # Initialize
        if hero_id not in self.observation:
            self.observation.update({hero_id: {}})
        
        # Remind equation in Python means symbol rather than assignment 
        hero = self.hero[hero_id]
        sensor_config = self.experiment_config["SENSOR_CONFIG"]
        observation_config = self.experiment_config["OBSERVATION_CONFIG"]

        # ---- Observation ---- 
        obs = self.observation[hero_id]
        if len(sensor_config["SENSOR"]) != len(observation_config["CAMERA_OBSERVATION"]):
            raise Exception("You need to specify the CAMERA_OBSERVATION for each sensor.")
        
        for i in range(0,len(sensor_config["SENSOR"])):
            if observation_config["CAMERA_OBSERVATION"][i]:
                obs['camera'].append(core.get_camera_data(hero_id))
        if observation_config["COLLISION_OBSERVATION"]:
            obs["collision"] = core.get_collision_data(hero_id)
        if observation_config["LOCATION_OBSERVATION"]:
            loc = hero.get_transform().location
            obs["location"] = [loc.x, loc.y, loc.z]
        if observation_config["LANE_OBSERVATION"]:
            obs["lane_invasion"] = core.get_lane_data(hero_id)
        if observation_config["GNSS_OBSERVATION"]:
            obs["gnss"] = core.get_gnss_data(hero_id)
        if observation_config["IMU_OBSERVATION"]:
            obs["imu"] = core.get_imu_data(hero_id)
        if observation_config["RADAR_OBSERVATION"]:
            obs["radar"] = core.get_radar_data(hero_id)
        if observation_config["BIRDVIEW_OBSERVATION"]:
            obs["birdview"] = core.get_birdview_data(hero_id)
        if observation_config["COMMUNICATION_OBSERVATION"]:
            # Get action and observation of nearby
            # filter = ['autopilot', 'hero']
            filter = ['autopilot']
            obs["communication"] = core.get_nearby_vehicles(hero_id, hero, max_distance=50, filter=filter)
        
        # ---- Navigation waypoint ---- 
        # TODO overcome the drawback caused by ergodic route search
        '''
        angle, distance = None, None
        while angle is None:
            angle, distance = self.get_closest_wp_forward(hero_id)

        obs["waypoint_angle"] = angle
        obs["waypoint_distance"] = distance
        self.past_distance_to_route[hero_id] = distance
        '''

        # ---- Info ---- 
        info = {}
        # ---- Physics control ----
        # action_taken = hero.get_control()
        action_taken = self.action[hero_id]
        info["action_taken"] = {
            "steer": action_taken.steer,
            "throttle": action_taken.throttle,
            "brake": action_taken.brake,
            "reverse": action_taken.reverse,
            "hand_brake": action_taken.hand_brake,
        }

        # ---- Intension of Carla Expert ----
        
        '''
        tm_port = 8000
        # traffic_manager = core.traffic_manager
        traffic_manager = core.client.get_trafficmanager(tm_port)
            
        # TODO make it stable to use
        # Returns next [INTENSION, Waypoint] traffic manager would choose
        # NOT physics vehicle control, Represented as ['LaneFollow', <carla.Waypoint>]
        action_possible = traffic_manager.get_all_actions(hero)
        action_next = traffic_manager.get_next_action(hero)
        
        info["action_possible"] = {
            'intension': action_possible[0],
            'waypoint': action_possible[1]
        }
        info["action_next"] = {
            'intension': action_next[0],
            'waypoint': action_next[1]
        }
        '''

        return obs, info

    def update_measurements(self, core):
        
        sensor_config = self.experiment_config["SENSOR_CONFIG"]
        observation_config = self.experiment_config["OBSERVATION_CONFIG"]
        if len(sensor_config["SENSOR"]) != len(observation_config["CAMERA_OBSERVATION"]):
                raise Exception("You need to specify the CAMERA_OBSERVATION for each sensor.")
        
        for i in range(0,len(sensor_config["SENSOR"])):
            if observation_config["CAMERA_OBSERVATION"][i]:
                core.update_camera(self.hero)
        if observation_config["COLLISION_OBSERVATION"]:
            core.update_collision(self.hero)
        if observation_config["LANE_OBSERVATION"]:
            core.update_lane_invasion(self.hero)

    # action from step is int
    # self.action stores carla.VehicleControl
    def update_actions(self, action):
        # add judgement for reset
        if action is None:
            for hero_id in self.hero:
                self.action.update({hero_id: carla.VehicleControl()}) # default：do nothing 
        else:
            for hero_id in self.hero:
                self.update_actions_single(action[hero_id], hero_id)

    def update_actions_single(self, action, hero_id):

        action = DISCRETE_ACTIONS[int(action)]

        _action = carla.VehicleControl()
        _action.throttle = float(np.clip(action[0], 0, 0.5))
        _action.steer = float(np.clip(action[1], -0.7, 0.7))
        _action.brake = float(np.clip(action[2], 0, 1))

        _action.reverse = action[3]
        _action.hand_brake = action[4]
        
        self.action.update({hero_id: _action})
        self.hero[hero_id].apply_control(_action)

    def compute_reward(self, core, observation, map):
        reward = {}
        for hero_id in self.hero:
            _reward = self.compute_reward_single(core, observation, map, hero_id)
            reward.update({hero_id: _reward})
        return reward
    
    def compute_reward_single(self, core, observation, map, hero_id):

        """
        :param core:
        :param observation:
        :return:
        """

        # ------------Reward------------
        reward = 0
        hero = self.hero[hero_id]
        obs = observation[hero_id]
        start_location = self.start_location[hero_id]

        speed_limit = self.speed_limit # default limit = 30kmh
        # ----Going farther-----
        # carla.Location.distance(): Returns Euclidean distance from this location to another one
        loc_hero = hero.get_location()
        vel_hero = self.get_speed(hero_id)
        distance_travelled = start_location.distance(loc_hero)
        reward += max(0, distance_travelled - self.previous_distance)
        self.previous_distance = max(self.previous_distance, distance_travelled)
        
        # ----Velocity is over speed limit by 10%----
        # Speed limit sign named as carla.traffic_sign.<speed_limit>
        sign_speed_limit = 120
        for actor in core.get_core_world().get_actors():
            loc_actor = actor.get_location()
            if loc_actor.distance(loc_hero) < 50 and 'speed_limit' in actor.type_id:
                sign_speed_limit = actor.type_id.split('.')[2] 
                break
        speed_limit = min(speed_limit, sign_speed_limit)
        
        if vel_hero <= 1.1 * speed_limit:
            reward += vel_hero
        else:
            reward -= vel_hero - 1.1 * speed_limit
        
        # TODO
        # ----Overtaking other vehicles----
        ###


        # ------------Penalty------------
        # ---- Collision & terminate immediately ----
        if obs["collision"]:
            reward -= 100
        
        # ---- Out of road & don't terminate ----
        # TODO add judgement of Solid/Broken Lane
        if obs["lane_invasion"]:
            reward -= 10
        
        # ---- Towards next waypoint ----
        # Penalty for deviating from the route 
        # TODO make it convincing
        # route_loss = obs["waypoint_distance"] - self.past_distance_to_route
        # reward -= max(0, route_loss/10)
        # self.past_distance_to_route = min(self.past_distance_to_route, obs["waypoint_distance"])
        
        # ---- TODO Traffic Light! ----
        ###

        return reward

    def initialize_reward(self, core):

        """
        Generic initialization of reward function
        :param core:
        :return:
        """
        print("This is a base experiment. Make sure you make you own reward initialization function")
        raise NotImplementedError


    # ==============================================================================
    # -------------------------- Hero -----------------------------------
    # ==============================================================================
    def spawn_hero(self, world, transform, autopilot=False, core=None):

        """
        This function spawns the hero vehicle. It makes sure that if a hero exists, it destroys the hero and respawn
        :param core:
        :param transform: Hero start location
        :param autopilot: Autopilot Status
        :return:
        """
        num_hero = self.experiment_config['n_heroes']
        self.spawn_points = world.get_map().get_spawn_points()

        self.hero_blueprints = world.get_blueprint_library().find(self.hero_model)
        self.hero_blueprints.set_attribute("role_name", "hero")

        self.hero = {}

        i = 0
        random.shuffle(self.spawn_points, random.random)
        num_hero_exist = 0
        server_port = core.tm_port
        # Keep finding spawn point until all self.hero are successfully spawned
        while num_hero_exist < num_hero:
            next_spawn_point = self.spawn_points[i % len(self.spawn_points)]
            hero = world.try_spawn_actor(self.hero_blueprints, next_spawn_point)
            if hero is not None:
                hero.set_autopilot(autopilot)# whether to use autopilot
                self.hero.update({hero.id: hero})
                self.start_location.update({hero.id: self.spawn_points[i].location})
                self.route.update({hero.id: self.select_random_route(world, hero.id)})
                self.t_idle_start.update({hero.id: time.time()})
                self.t_ep_start.update({hero.id: time.time()})
            else:
                print("Could not spawn Hero, changing spawn point")
                i+=1
            num_hero_exist = len(self.hero)

        world.tick()
        print("Hero all spawned!") # TODO realize respawn
        
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
        self.update_actions(action)

    
    def select_random_route(self, world, hero_id):
        '''
        retruns a random route for the car/veh
        out of the list of possible locations locs
        where distance is longer than 100 waypoints
        '''  
        hero = self.hero[hero_id]
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
    
    def get_closest_wp_forward(self, hero_id):
        '''
        this function is to find the closest point looking forward
        if there in no points behind, then we get first available
        '''

        # first we create a list of angles and distances to each waypoint
        # yeah - maybe a bit wastefull
        hero = self.hero[hero_id]
        vehicle_transform = hero.get_transform()
        points_ahead = []
        points_behind = []
        for i, wp in enumerate(self.route[hero_id]):
            #get angle  
            wp_transform = wp[0].transform
            distance = ((wp_transform.location.y - vehicle_transform.location.y)**2 + (wp_transform.location.x - vehicle_transform.location.x)**2)**0.5
            angle = math.degrees(math.atan2(wp_transform.location.y - vehicle_transform.location.y,
                                wp_transform.location.x - vehicle_transform.location.x)) -  vehicle_transform.rotation.yaw
            
            if angle > 360:
                angle = angle - 360
            elif angle < -360:
                angle = angle + 360

            if angle > 180:
                angle = -360 + angle
            elif angle < -180:
                angle = 360 - angle 

            if abs(angle) <= 90:
                points_ahead.append([i,distance,angle])
            else:
                points_behind.append([i,distance,angle])
        
        # now we pick a point we need to get angle to 
        if len(points_ahead) == 0:
            closest = min(points_behind, key=lambda x: x[1])
            if closest[2] > 0:
                closest = [closest[0], closest[1], 90]
            else:
                closest = [closest[0], closest[1], -90] 
        else:
            closest = min(points_ahead, key=lambda x: x[1])
            # move forward if too close
            for i, point in enumerate(points_ahead):
                if point[1] >= 10 and point[1] < 20:
                    closest = point
                    break
            return closest[2]/90.0, closest[1] # we convert angle to [-1 to +1] and also return distance
        
    def get_hero_location(self):
        loc = {}
        for hero_id in self.hero:
            _loc = self.hero[hero_id].get_location()
            loc.update({hero_id: _loc})
        return loc