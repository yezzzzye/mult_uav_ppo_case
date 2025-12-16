import gym
from gym import spaces
import copy
import numpy as np
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UAV_Imag_path = os.path.join(_project_root, "icon", "vector1.png")
Target_Imag_path = os.path.join(_project_root, "icon", "vector2.png")
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']}

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None,continuous_actions=False,done_callback=None,info_callback=None,
                  shared_viewer=True):

        self.world = world
        # 返回所有可以被训练的智能体对象
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        # 可以被训练的智能体个数
        self.n = len(world.policy_agents)

        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        # 动作空间离散化
        self.continuous_actions = continuous_actions
        self.discrete_action_space = not self.continuous_actions

        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0
        # configure spaces
        '''
        动作空间和观测空间
        '''
        # 智能体动作空间和观测空间
        share_obs_dim = 0
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)##共5个，这个space是env的属性和actor没有关系的，所以得到actor的输出后还需要转化为所需要的action
            else:##定义连续速度的动作空间
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_a,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c) ## 0和1
            else:##定义连续交流的动作空间
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            assert all([isinstance(act_space, spaces.Box) for act_space in total_action_space]) == True
            # action_space
            # 这里把交流不算做动作维度
            self.action_space.append(total_action_space[0])

            # observation space
            # 调用scenario的观测数据接口
            obs_dim = len(observation_callback(agent, self.world))                                                  ##观测数据维数（个数）
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))##obs_dim行的box
            share_obs_dim = obs_dim * self.n
        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.n)]


        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewer = None
        else:
            self.viewer = None * self.n
        self._reset_render()

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent)
        # advance world state
        # 同时执行world中的step
        self.world.step()
        # record observation for each agent
        # 记录智能体的历史观测数据
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))
            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case

        if self.shared_reward:
            reward = [np.sum(reward_n)]
            reward_n = [reward] * self.n
        return obs_n, reward_n, done_n, info_n

    # 环境里的初始化接口，返回初始的观测
    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        self._reset_render()
        # record observations for each agent，这里返回的是所有智能体的观测信息
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        action = np.array([action]) ##把采样得到的动作转化为列表  操作后，action[0]为具体动作

        # if agent.movable:##智能体可移动
        # physical action
        # 以下为每个智能体动作更新
        agent.action.u = action[0]
        sensitivity = 1.0
        if agent.accel is not None:
            sensitivity = agent.accel
        agent.action.u *= sensitivity##对动作进行放大
        action = action[1:]
        # if not agent.silent:
        #     # communication action
        #     agent.action.c = action[0]
        #     action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # render environment
    def render(self,mode='human'):
        from mpe._mpe_utils import rendering
        # 设定初始边界
        if self.viewer is None:
            self.viewer = rendering.Viewer(700, 700, caption="仿真平台")
        # create rendering geometry
        #   创建渲染几何体
        if self.render_geoms is None:
            self.render_geoms = []
            self.render_geoms_xform = []
            # 设置智能体图形
            for entity in self.world.entities:
                ##返回FilledPolygon()类型对象,为Geom()类型子类  ##设置颜色
                if 'uav' in entity.name:
                    geom = rendering.make_Image(UAV_Imag_path, 0.2, 0.2)
                    xform = rendering.Transform()  ##Transform()类型对象
                    # geom.set_color(*entity.color[:3], alpha=0.5)
                else:
                    geom = rendering.make_Image(Target_Imag_path, 0.3, 0.3)
                    xform = rendering.Transform()
                    # geom.set_color(*entity.color[:3])
                geom.add_attr(xform)

                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)
            # 障碍物图形
            for obstacle in self.world.obstacles:
                # 画的圆
                geom = rendering.make_circle(obstacle.size)  ##返回FilledPolygon()类型对象,为Geom()类型子类
                xform = rendering.Transform()                ##Transform()类型对象
                geom.set_color(*obstacle.color[:3], 0.3)     ##设置颜色，不透明度
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # 设置目标区域图形
            geom = rendering.make_circle(self.world.target_radius, 60, False)
            xform = rendering.Transform()
            geom.set_color(1, 0, 0)
            geom.add_attr(xform)
            self.render_geoms.append(geom)
            self.render_geoms_xform.append(xform)

            # add geoms to viewer
            self.viewer.geoms = []
            for geom in self.render_geoms:
                self.viewer.add_geom(geom)

        # 制智能体轨迹更新
        for i, agent in enumerate(self.world.agents):
            # 这里需要用深拷贝
            self.line[i].append(copy.deepcopy(agent.state.p_pos))
            geom = rendering.make_polyline(self.line[i])
            # Transform()类型对象
            xform = rendering.Transform()
            geom.set_color(*agent.color[:3])
            geom.add_attr(xform)
            self.viewer.add_geom(geom)

        # 设置智能体信号,原理同交流信息
        self.viewer.text_signal = []
        self.signal_xform = []
        for idx, entity in enumerate(self.world.entities + self.world.obstacles):
            signal = rendering.TextLine(self.viewer.window, idx)
            xform = rendering.Transform()
            signal.add_attr(xform)
            self.viewer.add_signal(signal)
            self.signal_xform.append(xform)
            # 载入信息
            if 'uav' in entity.name:
                word = entity.name + ' goal: ' + str(entity.state.goal)
                self.viewer.text_signal[idx].set_text(word, 10, True)
            elif 'target' in entity.name:
                self.viewer.text_signal[idx].set_text(str(entity.name), 10, True)
            else:
                self.viewer.text_signal[idx].set_text(str(entity.name), 10, True)

        # 设置交流信息,创建信息TextLine类加入viewer对象,利用TextLine类载入消息
        self.viewer.text_lines = []
        for idx, agent in enumerate(self.world.agents):
            if not agent.silent:
                info = rendering.TextLine(self.viewer.window, idx)
                self.viewer.add_text(info)
                word = agent.state.c
                # message = (agent.name + ' :goal:' + str(agent.state.goal) + '   info: ' + word + '   ')
                message = (agent.name + ' :goal:' + str(agent.state.goal))
                # 载入信息
                self.viewer.text_lines[idx].set_text(message, 12)

        # 渲染的场景更新，包括场景自适应缩放，以及位置更新
        # update bounds to center around agent
        all_poses = [entity.state.p_pos for entity in self.world.entities]  ##加入agents和landmark位置
        cam_range = np.max(np.abs(np.array(all_poses))) + 1
        self.viewer.set_max_size(cam_range)
        # update geometry positions  agents图形位置,signal
        all_pos = self.world.entities + self.world.obstacles
        for e, entity in enumerate(all_pos):
            if 'uav' in entity.name:
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
                self.render_geoms_xform[e].set_rotation(entity.state.yaw)
                self.signal_xform[e].set_translation(*entity.state.p_pos - [0, 0.2])
            else:
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
                self.signal_xform[e].set_translation(*entity.state.p_pos + [0, 0.2])
            self.signal_xform[e].set_scale(2 * cam_range / 700, 2 * cam_range / 700)

        # 更新target区域的位置
        self.render_geoms_xform[-1].set_translation(*self.world.target_centre)
        # render to display or array，渲染图形、信号、消息

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

        # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None
        # 轨迹数组 建议不用[[],]*n的方法
        self.line = []
        for _ in range(self.n):
            self.line.append([])

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self._reset_render()
