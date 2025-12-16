import torch
import os
import numpy as np
import pyglet
from torch.utils.tensorboard import SummaryWriter
import argparse
import matplotlib.pyplot as plt
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous
from mpe.MPE_env import MPEEnv


def evaluate_policy(args, env, agents,state_norm,seed=0):
    times = 2
    evaluate_reward = []
    for i in range(times):
        file_dir = args.save_dir + '/evaluate/seed_{}/eval_{}/'.format(seed,i)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        # auv_a = np.zeros((50, 6))
        # auv_v = np.ones((50, 6))*0.6
        # auv_w = np.zeros((50, 6))
        # auv_pesai = np.zeros((50, 6))
        # auv_path = np.zeros((50, 6))
        s = env.reset()
        episode_steps = 0
        dones = np.zeros(env.n)
        episode_rewards = np.zeros(env.n)
        #while (not np.all(dones)) and (episode_steps < args.max_episode_steps):
        while episode_steps < 35:
            episode_steps += 1
            # 收集所有智能体动作
            actions = []
            for agent_id in range(env.n):
                # 观测归一化
                # s = state_norm(s[agent_id])
                a = agents[agent_id].evaluate(s[agent_id])
                if args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
                else:
                    action = a
                actions.append(action)

                # auv_w[episode_steps][agent_id] = action[0]
                # auv_a[episode_steps][agent_id] = action[1]
            #print(actions)
            # 所有智能体与环境交互
            s_next, r, done, _ = env.step(actions)
            # 进行渲染,保存图片
            env.render()
            print(done)
            pyglet.image.get_buffer_manager().get_color_buffer().save(file_dir+f'step{episode_steps}.png')
            # time.sleep(0.05)
            for agent_id in range(env.n):
            # 计算累计奖励
                episode_rewards[agent_id] += r[agent_id]

                # auv_pesai[episode_steps][agent_id] = s_next[agent_id][0]
                # auv_v[episode_steps][agent_id] = s_next[agent_id][1]

            # 更新state
            s = s_next
            dones = done
        # env关闭清空
        #env.close()
        # 每一次eval的奖励加入
        evaluate_reward.append(episode_rewards)
    for i in range(times):
        print("eval_{}:episode rewards:{} agents rewards:{}".format(
            i,
            evaluate_reward[i].sum(),
            evaluate_reward[i]
        ))
    # fig, axs = plt.subplots(2, 2, figsize=(9, 9))
    # X = np.arange(episode_steps)
    # for agent_id in range(env.n):
    #     axs[0][0].set_title("pesai-steps"), axs[0][0].plot(X,auv_pesai[0:episode_steps,agent_id],label = f"uav_{agent_id}"),axs[0][0].legend(),axs[0][0].grid(True)
    #     axs[0][1].set_title("v-steps"),axs[0][1].plot(X,auv_v[0:episode_steps,agent_id],label = f"uav_{agent_id}"),axs[0][1].legend(),axs[0][1].grid(True)
    #     axs[1][0].set_title("w-steps"),axs[1][0].plot(X,auv_w[0:episode_steps,agent_id],label = f"uav_{agent_id}"),axs[1][0].legend(),axs[1][0].grid(True)
    #     axs[1][1].set_title("a-steps"),axs[1][1].plot(X,auv_a[0:episode_steps,agent_id],label = f"uav_{agent_id}"),axs[1][1].legend(),axs[1][1].grid(True)
    # plt.savefig(file_dir+'状态对比图.png')
    # plt.show()

    return np.sum(evaluate_reward)/times
def eval_main(args, seed):
    env = MPEEnv(args)

    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_space[0].shape[0]
    args.action_dim = env.action_space[0].shape[0]
    args.max_action = float(env.action_space[0].high[0])

    agents = []
    replay_buffers = []
    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    # 创建模型、算法、经验池
    for agent_id in range(env.n):
        replay_buffers.append(ReplayBuffer(args))
        agents.append(PPO_continuous(args))
    # 加载现有模型
    for agent_id in range(env.n):
        agents[agent_id].restore(agent_id)

    # eval
    evaluate_reward = evaluate_policy(args, env, agents, state_norm,seed)


def main(args, seed):
    env = MPEEnv(args)
    # env_evaluate = MPEEnv(args)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    args.state_dim = env.observation_space[0].shape[0]
    args.action_dim = env.action_space[0].shape[0]
    args.max_action = float(env.action_space[0].high[0])

    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))

    # Build a tensorboard
    log_dir = args.save_dir+'/train/PPO_continuous_{}/{}'.format(args.policy_dist,args.date)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    total_steps = 0  # Record the total steps during the training
    agents = []
    replay_buffers = []
    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    # 创建模型、算法、经验池
    for agent_id in range(env.n):
        replay_buffers.append(ReplayBuffer(args))
        agents.append(PPO_continuous(args))
    # 加载现有模型
    if args.restore:
        for agent_id in range(env.n):
            agents[agent_id].restore(agent_id)
    # if args.use_reward_norm:  # Trick 3:reward normalization
    #     reward_norm = Normalization(shape=1)
    # elif args.use_reward_scaling:  # Trick 4:reward scaling
    #     reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
    while total_steps < args.max_train_steps:
        s = env.reset()

        # if args.use_reward_scaling:
        #     reward_scaling.reset()

        episode_steps = 0
        dones = np.zeros(env.n)
        episode_rewards = np.zeros(env.n)
        while (not np.all(dones)) and (episode_steps < args.max_episode_steps):
            episode_steps += 1
            # 收集所有智能体动作和概率
            actions = []
            actions_logprob = []
            states = []
            states_next = []
            for agent_id in range(env.n):
                a, a_logprob = agents[agent_id].choose_action(s[agent_id])  # Action and the corresponding log probability
                if args.policy_dist == "Beta":
                    action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
                else:
                    action = a
                actions.append(action)
                actions_logprob.append(a_logprob)

            # 所有智能体与环境交互
            s_next, r, done, _ = env.step(actions)
            # 将各智能体信息存入经验池
            for agent_id in range(env.n):
                if done[agent_id] and episode_steps != args.max_episode_steps:
                    dw = True
                else:
                    dw = False
                replay_buffers[agent_id].store(s[agent_id], actions[agent_id], actions_logprob[agent_id], r[agent_id], s_next[agent_id], dw, done[agent_id])
                # 计算累计奖励
                episode_rewards[agent_id] += r[agent_id]
            # 更新智能体状态
            s = s_next
            total_steps += 1

        # train
        # When the number of transitions in buffer reaches batch_size,then update
        if replay_buffers[0].count == args.buffer_size:
            for agent_id in range(env.n):
                agents[agent_id].update(replay_buffers[agent_id], total_steps)
                replay_buffers[agent_id].count = 0

        # save
        if (total_steps//args.max_episode_steps) % args.save_freq == 0:
            for agent_id in range(env.n):
                agents[agent_id].save(agent_id,total_steps)

        # eval
        # Evaluate the policy every 'evaluate_freq' steps
        if (total_steps//args.max_episode_steps) % args.evaluate_freq == 0:
            evaluate_reward = evaluate_policy(args, env, agents,state_norm)
            writer.add_scalars("eval_episode_rewards/total_episodes", {"rewards/episodes": evaluate_reward}, total_steps//args.max_episode_steps)

        # log
        print("episodes:{} episode rewards:{} agents rewards:{}".format(
            total_steps // args.max_episode_steps,
            episode_rewards.sum(),
            episode_rewards
        ))
        writer.add_scalars("train_episode_rewards/total_steps", {"rewards/steps": episode_rewards.sum()}, total_steps)
        writer.add_scalars("train_episode_rewards/total_episodes", {"rewards/episodes": episode_rewards.sum()}, total_steps//args.max_episode_steps)



if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--scenario_name", type=str, default="simple_spread", help=" scenario_name")
    parser.add_argument("--date", type=str, default="2025_12_16", help="date")
    parser.add_argument("--max_episode_steps", type=int, default=50, help="max_episode_steps")

    parser.add_argument("--max_train_steps", type=int, default=int(7.6e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=500, help="Evaluate the policy every 'evaluate_freq' episodes")

    parser.add_argument("--restore", type=bool, default=True, help="restore or not")
    parser.add_argument("--save_freq", type=int, default=300, help="Save frequency")
    parser.add_argument("--save_dir", type=str, default="./data", help="save_dir")
    parser.add_argument("--model_dir", type=str, default="./data/best/model/300", help="model_dir")

    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")

    # 只用当前轮游戏数据更新，buffer_size应当等于max_episode_steps
    parser.add_argument("--buffer_size", type=int, default=1000, help="buffer_size")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size")

    parser.add_argument("--mini_batch_size", type=int, default=200, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=8.8e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=8.8e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    if torch.cuda.is_available():
        print("choose to use gpu...")
        args.device = torch.device("cuda:2")
    else:
        print("choose to use cpu...")
        args.device = torch.device("cpu")

    #main(args, seed=10)
    eval_main(args, seed=0)
