# PPO-Continuous for Multi-Agent Environments

这是一个基于 PyTorch 实现的连续动作空间 PPO（Proximal Policy Optimization）算法，专门用于多智能体(无人机/无人车)强化学习环境（MPE - Multi-Agent Particle Environment）。

##demo演示
https://github.com/user-attachments/assets/18e62483-6f76-4076-adca-c8647b036aaa
https://github.com/user-attachments/assets/ea270eb5-1421-49af-b803-a8f3bc656f87
https://github.com/user-attachments/assets/d8f98f63-9f6b-4217-b9ea-298e51aea309

## 特性

- ✅ 连续动作空间的 PPO 算法实现
- ✅ 多智能体环境支持
- ✅ 10 种训练技巧优化
- ✅ TensorBoard 训练可视化
- ✅ 自定义 MPE 环境（多无人机/无人车任务）

## 10 种训练技巧

1. **Advantage Normalization** - 优势函数归一化
2. **State Normalization** - 状态归一化
3. **Reward Normalization** - 奖励归一化
4. **Reward Scaling** - 奖励缩放
5. **Policy Entropy** - 策略熵正则化
6. **Learning Rate Decay** - 学习率衰减
7. **Gradient Clip** - 梯度裁剪
8. **Orthogonal Initialization** - 正交初始化
9. **Adam Optimizer Epsilon Parameter** - Adam 优化器参数设置
10. **Tanh Activation Function** - Tanh 激活函数

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- 其他依赖见 `requirements.txt`

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/yezzzzye/mult_uav_ppo_case.git
cd mult_uav_ppo_case
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

### 训练

直接运行训练脚本：
```bash
python train.py
```

### 自定义参数

可以通过命令行参数自定义训练配置：

```bash
python train.py \
    --scenario_name simple_spread \
    --max_train_steps 7600000 \
    --max_episode_steps 50 \
    --evaluate_freq 500 \
    --save_freq 300 \
    --policy_dist Gaussian \
    --lr_a 8.8e-4 \
    --lr_c 8.8e-4 \
    --gamma 0.99
```

### 主要参数说明

- `--scenario_name`: 场景名称（默认: `simple_spread`）
- `--max_train_steps`: 最大训练步数（默认: 7600000）
- `--max_episode_steps`: 每个回合最大步数（默认: 50）
- `--policy_dist`: 策略分布类型，`Gaussian` 或 `Beta`（默认: `Gaussian`）
- `--restore`: 是否加载已有模型（默认: `False`）
- `--save_dir`: 模型保存目录（默认: `./data`）
- `--model_dir`: 模型加载目录

## 项目结构

```
mult_uav_ppo_case/
├── train.py              # 主训练脚本
├── ppo_continuous.py     # PPO 算法实现
├── normalization.py      # 归一化工具
├── replaybuffer.py       # 经验回放缓冲区
├── demo                  # case演示
├── mpe/                  # 多智能体环境
│   ├── MPE_env.py        # MPE 环境封装
│   ├── environment.py    # 多智能体环境基类
│   └── scenarios/        # 场景定义
└── requirements.txt      # 依赖包列表
```

## 训练结果

训练过程中的日志会保存在 TensorBoard 中，可以通过以下命令查看：

```bash
tensorboard --logdir=./data/train
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 参考

- 原始 PPO 论文: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- 中文教程: [知乎专栏](https://zhuanlan.zhihu.com/p/512327050)
