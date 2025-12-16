import numpy as np
import math
import copy


class EntityState:  # physical/external base state of all entities
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


class AgentState(EntityState):  # state of agents (including communication and internal/mental state)
    def __init__(self):
        super().__init__()
        # communication utterance

        self.before_action_p_pos = None  # 更新动作前的位置
        self.before_action_p_vel = None  # 更新动作后的速度
        self.before_action_yaw = None  #   更新动作前的角度
        self.step_path_length = None  # 单步路径
        self.total_path_length = None  # 总路径长度
        self.yaw = None  # 角度
        self.goal_yaw = None
        self.c = None  # 交流
        self.goal = None  # 目标


class AgentAction:  # action of the agent
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


class LandmarkState(EntityState):
    def __init__(self):
        super().__init__()
        # communication utterance
        self.c = None


class Entity:  # properties and state of physical world entity
    def __init__(self):
        # name
        self.name = ''
        # properties:
        self.size = 2
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

class Obstacle(Entity):  # properties of landmark entities
    def __init__(self):
        super().__init__()

class Landmark(Entity):  # properties of landmark entities
    def __init__(self):
        super().__init__()
        self.state = LandmarkState()


class Agent(Entity):  # properties of agent entities
    def __init__(self):
        super().__init__()
        self.done = False
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = AgentAction()
        # script behavior to execute
        self.action_callback = None


class World:  # multi-agent world
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.Obstacles = []
        self.target_radius = 0.55
        self.target_centre = np.array([2, 0])
        # 控制量个数
        self.dim_a = 2
        # communication channel dimensionality
        self.dim_c = 1
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.1  # 阻力系数,默认0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def Landmarks(self):
        return self.landmarks
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # return all Obstacles
    @property
    def obstacles(self):
        return self.Obstacles

    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        # update agent action/state
        control = [None] * len(self.policy_agents)
        # True为有噪音,false为没有控制噪音,默认设置false
        control = self.apply_noise_into_control(control, False)
        # print(control)
        # True为有噪音,false为没有阻尼,默认设为false，damping_flag标志
        self.integrate_state(control, False)


    def apply_noise_into_control(self, control, noise_flag=False):
        # 把得到的每个智能体控制量赋给control
        for i, agent in enumerate(self.agents):
            if agent.movable and noise_flag is True:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                control[i] = agent.action.u + noise
            elif agent.movable and noise_flag is False:
                control[i] = agent.action.u
        return control

    # integrate physical state
    def integrate_state(self, control, damping_flag):
        for i, agent in enumerate(self.agents):
            if not agent.movable:
                continue
            # 保存上一个位置
            agent.state.before_action_p_pos = copy.deepcopy(agent.state.p_pos)
            agent.state.before_action_yaw = copy.deepcopy(agent.state.yaw)
            #agent.state.before_action_p_vel = copy.deepcopy(agent.state.p_vel)
            # print(agent.name, ":", agent.state.p_vel)
            # 阻尼情况
            if damping_flag is True:
                agent.state.p_vel = agent.state.p_vel * (1 - self.damping)
            if (control[i] is not None):
                self.control_vel_yaw(agent, control[i])

    # 运动学方程
    def kinematic_eqs(self, X, U):
        x_dot = X[2] * np.cos(X[3])
        y_dot = X[2] * np.sin(X[3])
        v_dot = U[0]
        pesai_dot = U[1]
        return np.array([x_dot.copy(), y_dot.copy(), v_dot.copy(), pesai_dot.copy()])

    def simple_kinematic_eqs(self):
        pass

    def rk2(self, f, x, u, dt):
        k1 = dt * f(x, u)
        k2 = dt * f(x + k1, u)
        return x.copy() + (k1 + k2) / 2

    def uav_model(self, agent, dt, r_action, accel_action, yaw_action):
        # 1.控制量
        a = accel_action    # 线加速度(m**2/s)
        w = r_action        # 角速度(rad/s)
        # 2.初始状态
        x0 = agent.state.p_pos[0]  # t0时刻x方向位置
        y0 = agent.state.p_pos[1]  # t0时刻y方向位置
        v0 = agent.state.p_vel     # t0时刻速度
        pesai0 = agent.state.yaw   # t0时刻航向角

        X = np.array([x0, y0, v0, pesai0])
        U = np.array([a, w])
        # 3.状态更新
        [x1, y1, v1, pesai1] = self.rk2(self.kinematic_eqs, X, U, dt)

        # 4.限幅
        # pesai1 = np.clip(pesai1, -math.pi, math.pi)     # 航向角限幅(rad)
        v1 = np.clip(v1, 0.6, 3)  # 速度限幅

        # 5.更新状态
        agent.state.p_pos[0] = x1
        agent.state.p_pos[1] = y1
        agent.state.yaw = pesai1
        agent.state.p_vel = v1
        # 6.更新总径长度
        if not agent.done:
            agent.state.step_path_length = np.sqrt(
                np.power(x1.copy() - x0.copy(), 2) + np.power(y1.copy() - y0.copy(), 2))
            agent.state.total_path_length += agent.state.step_path_length
        # 7.更新交流信息
        agent.state.c = f"total-path-length{round(agent.state.total_path_length,6)}"

    """
    控制变量 
    """

    def control_vel_yaw(self, agent, control):
        r_low = -math.pi
        r_high = math.pi
        r_action = r_low + (control[0] - 0) * ((r_high - r_low) / (1 - 0))
        r_action = np.clip(r_action, r_low, r_high)
        # 加速度为0
        accel_low = -2
        accel_high = 2
        accel_action = accel_low + (control[1] - (0)) * ((accel_high - accel_low) / (1 - 0))
        accel_action = np.clip(accel_action, accel_low, accel_high)
        # 更新状态
        self.uav_model(agent, self.dt, r_action, accel_action, yaw_action=None)

