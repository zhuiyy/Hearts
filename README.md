# Hearts AI (Transformer + PPO)

这是一个基于深度强化学习（Deep RL）的红心大战（Hearts）AI 项目。项目采用 **Transformer** 架构作为策略网络，结合了 **监督学习（Behavior Cloning）** 和 **PPO（Proximal Policy Optimization）** 算法进行训练，旨在打造一个既懂基本规则又能涌现高级战术（如“猪羊变色”、“逼出黑桃Q”）的智能 Agent。

## ✨ 核心特性

- **🧠 Transformer 架构**: 专为卡牌游戏设计的神经网络，能够处理手牌、牌桌状态和历史出牌记录。
- **🎓 双阶段训练**:
  1. **监督预训练 (Supervised Pretraining)**: 模仿内置的 "Expert" 专家策略，快速学会合法出牌和基础战术。
  2. **强化学习 (PPO)**: 通过自我博弈（Self-Play）和 PPO 算法进一步优化，探索超越专家的策略。
- **🕵️ 专家策略 (Expert Policy)**: 内置包含 "Piggy Hunting"（猎猪）逻辑的启发式 AI，懂得在低分段逼出黑桃 Q。
- **📊 实时可视化**: 提供 Jupyter Notebook 仪表盘，实时监控训练 Loss、胜率和分数曲线。
- **🎮 交互式展示**: `showcase.py` 允许你加载训练好的模型，观看 AI 对战或亲自下场切磋。
- **🚀 性能优化**: 支持 GPU 加速、自动显卡选择、向量化 Batch 处理。

## 📂 项目结构

| 文件 | 说明 |
| :--- | :--- |
| `train.py` | **训练主程序**。包含监督预训练和 PPO 训练循环，支持断点续传。 |
| `transformer.py` | **模型定义**。定义了 `HeartsTransformer` 网络结构。 |
| `strategies.py` | **策略库**。包含 `ExpertPolicy`（专家）、`Random`（随机）、`Min/Max` 等启发式策略。 |
| `game.py` | **游戏引擎**。核心规则实现，支持训练模式和对战模式。 |
| `showcase.py` | **展示脚本**。加载模型进行演示对战，支持人类玩家介入。 |
| `visualization.ipynb` | **监控面板**。读取日志并绘制训练曲线（Loss, Score）。 |
| `gpu_selector.py` | **工具**。自动检测并选择可用的 NVIDIA GPU。 |
| `data_structure.py` | **数据结构**。定义 Card, Suit, Rank 等基础类。 |

## 🚀 快速开始

### 1. 环境准备

确保安装了 Python 3.8+ 和 PyTorch。

```bash
pip install torch numpy matplotlib
```

### 2. 开始训练

运行训练脚本，程序会自动进行预训练和 PPO 训练。

```bash
python train.py
```

*训练过程中会生成 `training_log.json` 和模型权重 `hearts_model.pth`。*

### 3. 监控训练

在 VS Code 中打开 `visualization.ipynb` 并运行所有单元格。它会实时刷新图表，展示：

- **Score**: AI 的得分（越低越好）。
- **Policy Loss**: 策略网络的收敛情况。
- **Value Loss**: 价值网络的预测误差。

### 4. 观看演示

训练完成后（或使用现有模型），运行展示脚本：

```bash
python showcase.py
```

你可以选择让 AI 互搏，或者自己加入游戏。

## 🧠 算法细节

### 输入表示

模型接收 52 张牌的 One-hot 编码，结合位置编码（手牌、出牌区、历史记录），通过 Transformer Encoder 提取特征。

### 奖励设计 (Reward Shaping)

- **基础分**: 每吃一分红心扣分，吃黑桃 Q 大幅扣分。
- **全红奖励 (Shoot the Moon)**: 如果成功集齐所有红心和黑桃 Q，给予正向大奖励。
- **中间奖励**: 每一墩（Trick）结算时给予即时反馈，加速收敛。

### 专家策略 (Expert Policy)

我们在 `strategies.py` 中实现了一个强大的规则型 AI，它懂得：

- **Piggy Hunting**: 当持有低点数黑桃且黑桃 Q 未出时，主动出黑桃逼迫对手出 Q。
- **Voiding Suits**: 尽快出完某个花色以便垫牌。
- **Safety First**: 在危险局面下优先出小牌。

---

*Created by zhuiyy*
