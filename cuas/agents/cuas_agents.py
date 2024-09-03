import math
import os
import pathlib
import numpy as np
from cuas import util
import math
from enum import IntEnum

path = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

# RESOURCE_FOLDER = pathlib.Path("resources")
RESOURCE_FOLDER = path.joinpath("../../resources")

class AgentType(IntEnum):
    P = 0  # agent
    #E = 1  # evader
    O = 2  # obstacle
    T = 3  # Target


class ObsType(IntEnum):
    S = 0  # Static
    M = 1  # Moving


class ObsPaddingType(IntEnum):
    last = 0
    zeros = 1
    max = 2
    min = 3


# TODO: move env_width and height to cuas_agents
class Entity:
    """Defines an entity for the simulation. All units are in metric"""

    def __init__(self, x, y, r=1, env_width_x=200, env_height_y=200, type2=AgentType.P):
        self.x = x
        self.y = y
        self.radius = r
        self.env_width_x = env_width_x
        self.env_height_y = env_height_y
        self.env_xgrid_res = 2
        self.env_ygrid_res = 2
        self._in_collision = False
        self._type = type2
        #self._state = np.array([self.x, self.y, self.radius])
        if type2 == AgentType.P: 
            y = y.item()
            self.y = y 
        self._state = np.array([self.x, self.y, self.radius])
        
    @property
    def type2(self):
        return self._type

    @type2.setter
    def type2(self, type2):
        self._type = type2

    @property
    def pos(self):
        """Return agent's position

        Returns:
            _type_: _description_
        """
        return np.array([self.x, self.y])

    def rel_dist(self, entity):
        """Calculates relative distance to another object"""
        dist = util.distance((self.x, self.y), (entity.x, entity.y))

        return dist

    def rel_dist2(self, entity_x,entity_y):
        """Calculates relative distance to another object"""
        dist = util.distance((self.x, self.y), (entity_x, entity_y))

        return dist

    #def rel_flux(self,entity,rho,ws): 
    #    #alpha = util.angle((self.x, self.y), (entity.x, entity.y))



    #    alpha = math.atan(self.y - entity.y, self.x - entity.x) 
        
    #    xpt = int(np.floor(entity.x/self.env_xgrid_res))
    #    ypt = int(np.floor(entity.y/self.env_ygrid_res))

    #    xpt, ypt, b_error = self.check_points(xpt,ypt)

    #    if b_error == True: 
    #        Fij = 0
    #        return Fij 

    #    rho_j = rho[xpt,ypt]

    #    ypt2 = [3*ypt, 3*ypt + 1]

    #    V_j = ws[xpt,ypt2]

    #    Fij = rho_j * np.linalg.norm(V_j) * cos(math.atan(V_j[1]/V_j[0]) - alpha)

    #    return Fij

    def concentration(self,rho): 
        xpt = int(np.floor(self.x/self.env_xgrid_res))
        ypt = int(np.floor(self.y/self.env_ygrid_res))

        xpt, ypt, b_error = self.check_points(xpt,ypt)

        if b_error == True: 
            rho_i = 0 
            return rho_i 
        #else: 
        rho_i = rho[xpt,ypt]

        #print('Need to update concentration in cuas_agents.py')
        #quit()

        return rho_i

    #def check_points(self,x,y): 
    #    if x >= 100: 
    #        x = 99
    #    if y >= 100: 
    #        y = 99
    #    if x < 0: 
    #        x = 0
    #    if y < 0: 
    #        y = 0

    #    return x, y

    def check_points(self,x,y): 
        b_error = False 
        if x == 100: 
            x = 99
        if y == 100: 
            y = 99
        if x < 0 or x > 100: 
            b_error = True 
        if y < 0 or y > 100: 
            b_error = True 

        return x, y, b_error



    def wind_coor(self,ws): 
        xpt = int(np.floor(self.x/self.env_xgrid_res))

        ypt = int(np.floor(self.y/self.env_ygrid_res))

        xpt, ypt, b_error = self.check_points(xpt,ypt)

        if b_error == True: 
            #V_i = [0 ,0, 0] 
            V_i = [0, 0]
            return V_i 
        
        # Use for 3D wind coordinates 
        #ypt2 = [3*ypt, 3*ypt + 1, 3*ypt + 2]
        # Use for 2D wind coordinates 
        ypt2 = [3*ypt, 3*ypt + 1]

        V_i = ws[xpt,ypt2]

        #print('Need to update wind_coor in cuas_agents.py')
        #quit()

        return V_i

    #def flux(self,entity,rho,ws,cent): 
    #    #alpha = self.theta - rel_bearing_error(entity)
    #    xpt = int(np.floor(entity.x/self.env_xgrid_res))
    #    ypt = int(np.floor(entity.y/self.env_ygrid_res))

    #    xpt, ypt, b_error = self.check_points(xpt,ypt)

    #    if b_error == True: 
    #        Fi = 0
    #        return Fi

    #    #ypt2 = [3*ypt, 3*ypt + 1, 3*ypt + 2]
    #    ypt2 = [3*ypt, 3*ypt + 1]

    #    rho_i = rho[xpt,ypt]
    #    V_i = ws[xpt,ypt2]

    #    n = np.array([xpt,ypt]) - cent

    #    #Fi = (rho_i - 1.98)*(1e-6)*(101325*16.04/(8.3145*273))*np.linalg.norm(np.dot(V_i,[0,0,1])) * 4 
    #    Fi = (rho_i - 1.98)*(1e-6)*(101325*16.04/(8.3145*273))*np.dot(V_i,n) * math.pi * 1**2

    #    #print('Need to update flux in cuas_agents.py')
    #    #quit()

    #    #Fi = rho_i * np.linalg.norm(np.dot(V_i,[0,0,1])) 

    #    return Fi


    def rel_bearing(self, entity):
        """Calculates relative bearing to another object"""
        bearing = util.angle((self.x, self.y), (entity.x, entity.y))

        return bearing

    def rel_bearing_error(self, entity):
        """[summary]

        Args:
            entity ([type]): [description]

        Returns:
            [type]: [description]
        """
        bearing = util.angle((self.x, self.y), (entity.x, entity.y)) - self.theta
        # TODO: verify this from Deep RL for Swarms
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        return bearing

    def rel_bearing_error2(self, entity_x,entity_y):
        """[summary]

        Args:
            entity ([type]): [description]

        Returns:
            [type]: [description]
        """
        bearing = util.angle((self.x, self.y), (entity_x, entity_y)) - self.theta
        # TODO: verify this from Deep RL for Swarms
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        return bearing

    def rel_bearing_entity_error(self, entity):
        bearing = util.angle((self.x, self.y), (entity.x, entity.y)) - entity.theta
        # TODO: verify this from Deep RL for Swarms
        bearing = (bearing + np.pi) % (2 * np.pi) - np.pi
        return bearing

    def collision_with(self, entity):
        """Returns if has collided with another entity"""
        in_collision = False

        rel_dist_to_entity = self.rel_dist(entity)

        #print(rel_dist_to_entity)
        #quit()

        if rel_dist_to_entity < (self.radius + entity.radius):
            in_collision = True
            self._in_collision = in_collision

        return in_collision


class Agent(Entity):
    """Defines an agent for the simulation. All units are in metric, time is second

    Args:
        Entity ([type]): [description]

    Returns:
        [type]: [description]
    """

    def __init__(
        self, x, y, theta, r=1, type2=AgentType.P, obs_r=20
    ):  # assume env size 80x60
        super().__init__(x, y, r=r, type2=type2)
        self.theta = theta  # in radians
        self.dt = 0.05  # 50 ms
        self.v = 0
        self.w = 0
        self.obs_r = obs_r

    # TODO: this should just return the states not the state variable
    @property
    def state(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return np.array([self.x, self.y, self.theta, self.v, self.w])

    def step(self, action):
        """Updates the agent's state. Positive radians is counterclockwise"""
        v = action[0]
        w = action[1]


        if w >= 0.01 or w <= -0.01:
            ratio = v / w
            self.x += -ratio * math.sin(self.theta) + ratio * math.sin(
                self.theta + w * self.dt
            )
            self.y += ratio * math.cos(self.theta) - ratio * math.cos(
                self.theta + w * self.dt
            )
            self.theta += w * self.dt

        else:
            self.x += v * self.dt * math.cos(self.theta)
            self.y += v * self.dt * math.sin(self.theta)
            self.theta += w * self.dt

        self.v = action[0]
        self.w = action[1]

        self._check_bounds()

    def _check_bounds(self):
        """Checks bound for moving objects"""

        self.x = min(max(0, self.x), self.env_width_x)
        self.y = min(max(0, self.y), self.env_height_y)

        # see: https://stackoverflow.com/questions/15927755/opposite-of-numpy-unwrap
        # self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        # wrap theta to -pi and pi
        # This is more efficient than using np.arctan2
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

        # TODO: fix this

    def sensed(self, entity):
        """Returns True if agent can sense entity
        # https://github.com/Attila94/EKF-SLAM/blob/master/robot.py
        # https://www.geeksforgeeks.org/check-whether-point-exists-circle-sector-not/
        # https://stackoverflow.com/questions/13652518/efficiently-find-points-inside-a-circle-sector
        """

        _detected = False
        rel_bearing = self.rel_bearing(entity)
        rel_distance = self.rel_dist(entity)
        return True if rel_distance < self.obs_r else False

    #def sensed_plume(self, entity,rho,ws,flux_threshold):
    #    """Returns True if agent can sense plume
    #    # https://github.com/Attila94/EKF-SLAM/blob/master/robot.py
    #    # https://www.geeksforgeeks.org/check-whether-point-exists-circle-sector-not/
    #    # https://stackoverflow.com/questions/13652518/efficiently-find-points-inside-a-circle-sector
    #    """

    #    _detected = False
    #    #rel_bearing = self.rel_bearing(entity)
    #    #rel_distance = self.rel_dist(entity)
    #    ff = self.flux(entity,rho,ws)
    #    det = True if ff >= flux_threshold else False
    #    return [det,ff]

    def get_detect(self,data,flux_threshold,cent): 
        xpt = int(np.floor(self.x/self.env_xgrid_res))

        ypt = int(np.floor(self.y/self.env_ygrid_res))

        xpt, ypt, b_error = self.check_points(xpt,ypt)

        if b_error == True: 
            Fi = 0
            rho_i = 0
            det = False 
            print(xpt)
            print(ypt)
            quit()
            return [det,Fi,rho_i]

        rho_i = data[0]
        V_i = data[1:3] 
        agent_loc = np.array([self.x,self.y])
        
        n_i = agent_loc - cent
        mag_n_i = np.linalg.norm(n_i)

        if mag_n_i == 0: 
            n_i = np.array([0,0])
        else: 
            n_i = n_i/mag_n_i 

        #n_i = np.array([math.cos(self.theta),math.sin(self.theta)])
        Fi = (rho_i - 1.98)*(1e-6)*(101325*16.04/(8.3145*273))*np.dot(V_i,n_i) * math.pi * 1**2
        #det = True if abs(Fi) >= flux_threshold else False

        det = True if (rho_i - 1.98) >= flux_threshold else False

        return [det,Fi,rho_i-1.98,V_i] 

    def uni_to_si_dyn(self, dxu, projection_distance=0.05):
        """
        See:
        https://github.com/robotarium/robotarium_python_simulator/blob/master/rps/utilities/transformations.py

        """
        cs = np.cos(self.theta)
        ss = np.sin(self.theta)

        dxi = np.zeros(2)
        dxi[0] = cs * dxu[0] - projection_distance * ss * dxu[1]
        dxi[1] = ss * dxu[0] + projection_distance * cs * dxu[1]

        return dxi

    def si_to_uni_dyn(self, si_v, projection_distance=0.05):
        """
        see:
        https://github.com/robotarium/robotarium_python_simulator/blob/master/rps/utilities/transformations.py
        also:
            https://arxiv.org/pdf/1802.07199.pdf

        Args:
            agent ([type]): [description]
            si_v ([type]): [description]
            projection_distance (float, optional): [description]. Defaults to 0.05.

        Returns:
            [type]: [description]
        """
        cs = np.cos(self.theta)
        ss = np.sin(self.theta)

        dxu = np.zeros(2)
        dxu[0] = cs * si_v[0] + ss * si_v[1]
        dxu[1] = (1 / projection_distance) * (-ss * si_v[0] + cs * si_v[1])

        return dxu


class Agent2D(Agent):
    def __init__(
        self, x, y, theta, r=1, type2=AgentType.P, obs_r=20
    ):  # assume env size 80x60
        super().__init__(x=x, y=y, theta=theta, r=r, type2=type2, obs_r=obs_r)
        self.dt = 0.05  # 10 ms
        self.vx = 0
        self.vy = 0

    @property
    def state(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return np.array([self.x, self.y, self.theta, self.vx, self.vy])

    def step(self, action):
        """Updates the agent's state. Positive radians is counterclockwise"""
        self.vx = action[0]
        self.vy = action[1]

        self.x += self.vx * self.dt
        self.y += self.vy * self.dt
        self.theta = np.arctan2(self.vy, self.vx)

        self._check_bounds()


class Obstacle(Agent):
    """Defines and obstacle in the environment
    Args:
        Agent ([type]): [description]
    """

    def __init__(
        self,
        x,
        y,
        theta,
        r=4,
        type2=AgentType.O,
        obs_type=ObsType.S,
        v_min=0,
        v_max=5,
        w_min=-np.pi,
        w_max=np.pi,
    ):
        super().__init__(x, y, theta, r, type2=type2)

        self._obs_type = obs_type
        self._v_min = v_min
        self._v_max = v_max
        self._w_min = w_min
        self._w_max = w_max

    def step(self):
        if self._obs_type == ObsType.S:
            action = [0, 0]

        else:
            # get random v and random w
            v = np.random.uniform(low=self._v_min, high=self._v_max)
            w = np.random.uniform(low=self._w_min, high=self._w_max)
            action = [v, w]

        super().step(action)

    # @overload
    # def _check_bounds(self):
    #     # min_x = -self.radius / 2
    #     # min_y = -self.radius / 2
    #     # max_x = self.env_width + self.radius / 2
    #     # max_y = self.env_height + self.radius / 2
    #     min_x = 0
    #     min_y = 0
    #     max_x = self.env_width
    #     max_y = self.env_height

    #     if self.x < min_x:
    #         self.x = max_x
    #     elif self.x > max_x:
    #         self.x = min_x
    #     if self.y < min_y:
    #         self.y = max_y
    #     elif self.y > max_x:
    #         self.y = min_y

    #     self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi


class CuasAgent(Agent):
    """[summary]

    Args:
        Agent ([type]): [description]
    """

    def __init__(self, id, type2, x, y, theta, r=1, obs_r=20, move_type="rl"):
        super().__init__(x, y, theta, r=r, obs_r=obs_r, type2=type2)
        self.id = id
        self.done = False
        self.reported_done = False
        self.move_type = move_type
        self.captured = False


class CuasAgent2D(Agent2D):
    """[summary]

    Args:
        Agent ([type]): [description]
    """

    def __init__(self, id, type2, x, y, theta, r=1, obs_r=20, move_type="rl"):
        super().__init__(x, y, theta, r=r, obs_r=obs_r, type2=type2)
        self.id = id
        self.done = False
        self.reported_done = False
        self.move_type = move_type
        self.captured = False