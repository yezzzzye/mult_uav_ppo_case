import numpy as np
from mpe._mpe_utils.core import Agent, Landmark,Obstacle, World
from mpe._mpe_utils.scenario import BaseScenario
from mpe.goal_match import match
RAD2DEG = 57.29577951308232
def obstacle_points(n,X,Range,var=None):
    if n % 2 == 0:
        # 画n为偶数系列1开始点
        n1 = n // 2
        x = X * np.ones(n) + np.random.normal(0,var,n)
        y2 = np.zeros(n1)
        for i in range(n1):
            y2[i] = -(i + 1) * Range / (n / 2) + Range / n
        y = np.concatenate([y2] + [np.flipud(-y2)]) + np.random.normal(0,var,n)
        points = np.c_[x, y]
    else:
        # 画n为奇数开始点
        x = X * np.ones(n) + np.random.normal(0,var,n)
        y1 = np.array([0]) + np.random.normal(0,var)
        y2 = np.zeros(n // 2)
        for i in range(n // 2):
            y2[i] = -(i + 1) * Range / (n // 2)
        y = np.concatenate([y1] + [y2] + [np.flipud(-y2)]) + np.random.normal(0,var,n)
        # 按行连接，左右相加，要求行数相同
        points = np.c_[x, y]
    return points

def start_points(n,r,point,var=None):
    x_star = point[0]-2.65
    x_end = point[0]-r/2.2
    """
    (x-point[0])**2+y**2 =  r**2
    y = kx + b
    """
    if n % 2 == 0:
        # 画n为偶数系列
        x1 = np.linspace(x_star, x_end, n // 2) + np.random.normal(0, var, n // 2)
        x2 = np.linspace(x_star, x_end, n // 2) + np.random.normal(0, var, n // 2)
        y1 = np.sqrt(abs(r ** 2 - (x1 - point[0]) ** 2)) + np.random.normal(0, var, n // 2)
        y2 = -np.sqrt(abs(r ** 2 - (x2 - point[0]) ** 2)) + np.random.normal(0, var, n // 2)
        #
        # x1 = np.random.normal(x_star, var, n // 2)
        # x2 = np.random.normal(x_star, var, n // 2)
        # y1 = np.linspace(0, 3, n // 2, endpoint=False) + np.random.normal(0, var, n // 2)
        # y2 = -np.linspace(0, 3, n // 2, endpoint=False) + np.random.normal(0, var, n // 2)
        x = np.r_[x1,x2]
        y = np.r_[y1,y2]

        points = np.c_[x, y]
    else:
        x0 = -r+np.random.normal(0, var)
        y0 = point[1] + np.random.normal(0, var)
        x1 = np.linspace(x_star, x_end, n // 2) + np.random.normal(0, var, n // 2)
        x2 = np.linspace(x_star, x_end, n // 2) + np.random.normal(0, var, n // 2)
        y1 = np.sqrt(abs(r ** 2 - (x1 - point[0]) ** 2)) + np.random.normal(0, var, n // 2)
        y2 = -np.sqrt(abs(r ** 2 - (x1 - point[0]) ** 2)) + np.random.normal(0, var, n // 2)
        x = np.r_[x0,x1,x2]
        y = np.r_[y0,y1,y2]
        # 按行连接，左右相加，要求行数相同
        points = np.c_[x, y]
    return points
def circle_points(n,center,r):
    if n % 2 == 0:
        # n为偶数情况下画目标点
        t = np.linspace(-np.pi + np.pi / n, np.pi - np.pi / n, n)
        x = r * np.cos(t)+center[0]
        y = r * np.sin(t)+center[1]
    else:
        # n为奇数情况下画目标点
        t = np.linspace(-np.pi, np.pi, n, endpoint=False)
        x = r * np.cos(t)+center[0]
        y = r * np.sin(t)+center[1]
    points = np.c_[x, y]
    return points

class Scenario(BaseScenario):
    def make_world(self, N = 3):
        # set any world properties first
        self.num_agents = N
        self.num_landmarks = N
        self.num_obstacles = 0
        world = World()
        world.collaborative = False
        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = f'uav_{i}'
            agent.collide = True
            agent.movable = True
            agent.silent = False
            agent.size = 0.05
        # add landmarks
        world.landmarks = [Landmark() for i in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'target %d' % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.15

        # add Obstacles
        world.Obstacles = [Obstacle() for i in range(self.num_obstacles)]
        for i, obstacle in enumerate(world.Obstacles):
            obstacle.name = 'obstacle %d' % i
            obstacle.collide = False
            obstacle.movable = False
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
            # movable
            agent.movable = True
            agent.done = False
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])

        # # 障碍物设置
        # temp = np.concatenate((obstacle_points((self.num_obstacles - 2) // 2, -0.3, 0.5, var=0.12),
        #                        obstacle_points(self.num_obstacles - 2 - (self.num_obstacles - 2) // 2, 0.9, 2.2,var=0.12)))
        #
        # self.obstaclePoints = np.concatenate((temp, obstacle_points(2, 2.9, 2.6, var=0.12)))
        for i, obstacle in enumerate(world.Obstacles):
            obstacle.size = np.random.uniform(4, 4.2)/10
            obstacle.color = np.array([1, 0, 0])
            # 随机障碍物位置
            # (x-2)**2+(y-0)**2 = 1.8**2
            obstacle.state.p_pos = self.obstaclePoints[i]

        self.startPoints = start_points(self.num_agents, 3, [0, 0], var=0.06)
        # set random initial states
        for i, agent in enumerate(world.agents):
            # 随机位置
            agent.state.p_pos = self.startPoints[i]
            #agent.state.before_action_p_pos = self.startPoints[i]
            # 速度和角度
            agent.state.p_vel = 0.6
            agent.state.yaw = 0
            # 路径长度
            agent.state.step_path_length = 0
            agent.state.total_path_length = 0
            # 交流信息
            agent.state.c = f"step_path_length{agent.state.step_path_length},total_path_length{agent.state.total_path_length}"

        self.circlePoints = circle_points(self.num_landmarks, world.target_centre, world.target_radius)
        for i, landmark in enumerate(world.landmarks):
            # 随机相对固定位置
            # 在圆上点
            landmark.state.p_pos = self.circlePoints[i]

        # yaw目标 
        goal_yaws = [0,np.pi / (3/2),-np.pi / (3/2)]
        # 计算目标匹配结果
        match_result = match(self.startPoints,self.circlePoints)
        for i in range(self.num_agents):
            # 选择目标
            goal_index = match_result[i]
            world.agents[i].state.goal = goal_index
            world.agents[i].state.goal_yaw = goal_yaws[goal_index]

    # 总奖励
    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        # 智能体已经done不累计加分
        if agent.done:
            return rew
        if agent.collide:
            agents_pos = np.array([a.state.p_pos for a in world.agents])
            ref_pos = np.tile(agent.state.p_pos, (self.num_agents,1))
            dists = np.sqrt(np.sum(np.square(agents_pos-ref_pos),axis=1))
            # 去掉零元素
            dists = dists[dists != 0]
            # 任意距离小于两倍智能体半径，即判断为碰撞
            if np.less(dists,2 * agent.size).any():
                rew -= 5
        rew += self.reward_arrived(agent,world)+self.reward_goal_agnet(agent,world) +self.path_length(agent, world)

        return rew

    # 单智能体靠近目标奖励
    def reward_goal_agnet(self,agent,world):
        rew = 0
        index = agent.state.goal
        l = world.landmarks[index]
        dists_BeforeAction = np.sqrt(np.sum(np.square(agent.state.before_action_p_pos - l.state.p_pos)))
        dists_AfterAction = np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))
        if(dists_AfterAction <= 0.6):
            errorbefore_action = abs(agent.state.before_action_yaw - agent.state.goal_yaw)
            errorafter_action = abs(agent.state.yaw - agent.state.goal_yaw)
            rew +=  (errorbefore_action - errorafter_action) * 5 
        else:
            rew +=   (dists_BeforeAction - dists_AfterAction) * 30

        return rew

    # 同时到达奖励
    def path_length(self, agent, world):
        rew = 0
        # 找到单个智能体的目标landmark
        index = agent.state.goal
        l = world.landmarks[index]
        ref_dis = np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))

        goal_path_AfterAction = np.zeros((self.num_agents))

        for i, a in enumerate(world.agents):
            # 找到单个智能体的目标landmark
            index = a.state.goal
            l = world.landmarks[index]
            goal_path_AfterAction[i] = np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))

        std_AfterAction = np.std(goal_path_AfterAction)*ref_dis/np.sum(goal_path_AfterAction)

        rew -= 10*std_AfterAction
        return rew

    # 单智能体到达目的地和经过危险区域奖励
    def reward_arrived(self,agent,world):
        rew = 0
        if (self.arrive_target(agent, world)):                  ##到达目标点
            agent.done = True                                                       ##引入done不可运动
            agent.movable = False                                         
            if(agent.state.goal_yaw-np.pi/6.5<= agent.state.yaw <=agent.state.goal_yaw+np.pi/6.5):
                agent.color = np.array([0.01, 0.99, 0.01])                          ##变成绿色                                               
                rew += 50
            else:
                agent.color = np.array([1, 0, 0])                                   ##变成红色
                rew -= 50
            return rew
        if (self.arrive_region(agent, world)):                  ##进入危险区域
            agent.color = np.array([1, 0, 0])                   ##变成红色
            rew -= 5
        else:
            agent.color = np.array([0.35, 0.35, 0.85])          ##变回原色
        return rew

    # 判断单个智能体是否到达目标点
    def arrive_target(self,agent,world):
        index = agent.state.goal
        l = world.landmarks[index]
        delta_pos = agent.state.p_pos - l.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent.size + l.size/2
        return True if dist <= dist_min else False

    # 判断单个智能体是否进入危险（红圈）、雷达扫描区域
    def arrive_region(self, agent,world):
        # 判断是否到达危险区
        # Obstacles_pos = np.array([O.state.p_pos for O in world.Obstacles])
        # Obstacles_size = np.array([O.size for O in world.Obstacles])
        # ref_pos = np.tile(agent.state.p_pos, (self.num_obstacles, 1))
        # dists = np.sqrt(np.sum(np.square(Obstacles_pos - ref_pos), axis=1))
        # if np.less(dists,Obstacles_size).any():
        #     return True
        # 判断是否到达雷达扫描区
        dist = np.sqrt(np.sum(np.square(agent.state.p_pos - world.target_centre)))
        if dist < world.target_radius:
            return True
        return False


    def observation(self, agent, world):
    #1 uav全局观测量
        #1.1 uav状态量
        agent_self_state=[]
        agent_self_state.append(agent.state.yaw)                    # 偏航角
        agent_self_state.append(agent.state.p_vel)                  # 速度
        # #1.2 相对目标点观测量
        # l = world.landmarks[agent.state.goal]
        # agent_self_state.append(np.sqrt(np.sum(np.square(agent.state.p_pos - l.state.p_pos)))) # 相对目标点距离
        # agent_self_state.append(agent.state.yaw - agent.state.goal_yaw)                        # 相对目标yaw

        #1.3 其他智能体相对目标距离
        other_dists = []
        for a in world.agents:
            if a == agent: continue
            # 找到单个智能体的目标landmark
            index = a.state.goal
            l = world.landmarks[index]
            other_dists.append(np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))))
        agent_self_state.append(np.mean(other_dists))

    #2 uav局部观测量
        cannon_pos = []
        # 2.0相对目标点
        l = world.landmarks[agent.state.goal]
        cannon_pos.append(l.state.p_pos-agent.state.p_pos)            
        # # 2.1障碍区观测
        # ref_pos = np.tile(agent.state.p_pos, (self.num_obstacles, 1))
        # dists = np.sqrt(np.sum(np.square(self.obstaclePoints - ref_pos), axis=1))
        # # 取前最小的
        # index = dists.argsort()[0]
        # cannon_pos.append(self.obstaclePoints[index] - agent.state.p_pos)           # 第一个相对禁戒区位置
        # 2.2扫描区观测
        cannon_pos.append(world.target_centre-agent.state.p_pos)                    # 相对雷达扫描区位置

        # 2.3相对其他智能体观测
        other_pos = []
        agents_pos = np.array([a.state.p_pos for a in world.agents])
        ref_pos = np.tile(agent.state.p_pos, (self.num_agents, 1))
        dists = np.sqrt(np.sum(np.square(agents_pos - ref_pos), axis=1))
        # 取前三个最小的，排除掉第一个自己0，剩余两个加入观测
        indexs = dists.argsort()[:3]
        other_pos.append((agents_pos[indexs[1]] - agent.state.p_pos)/5)             # 相对第一个其他智能体位置
        other_pos.append((agents_pos[indexs[2]] - agent.state.p_pos)/5)             # 相对第二个其他智能体位置

        return np.concatenate([agent_self_state] + cannon_pos + other_pos)
    # 单个智能体done
    def done(self,agent,world):
        return agent.done

