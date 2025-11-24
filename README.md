# Hearts AI (Transformer + PPO + DAgger)

这是一个基于深度强化学习（Deep RL）的红心大战（Hearts）AI 项目。项目采用 **Transformer** 架构作为策略网络，结合了 **DAgger (Dataset Aggregation)** 算法进行监督预训练，并通过 **PPO (Proximal Policy Optimization)** 算法进行强化学习微调。

经过精心设计的课程学习（Curriculum Learning），该 AI 已经从最初的随机出牌进化为能够与专家级对手抗衡的高手。

## 🏆 训练成果

*   **Pretrain 阶段**: 
    *   采用 **DAgger** 算法，不仅模仿 Expert 的出牌，还模仿其传牌策略（包括识别 Shooting the Moon 的机会）。
    *   在 5000 局预训练后，模型对 Expert 行为的预测准确率达到 **60%+**，Loss 降至 **1.58** 左右。
    *   模型成功学会了游戏规则、花色跟随、垫牌逻辑以及基础的攻防策略。
*   **PPO 阶段**:
    *   引入 **Entropy Decay** 机制，前期鼓励探索，后期专注于策略收敛。
    *   实施 **Curriculum Learning**，对手从 Random Bots 逐步升级为 Expert Bots，最后进行自我博弈。
    *   **当前战绩**: 在面对 3 个 Expert 对手时，模型的平均得分稳定在 **9.4 分**左右（平均分为 6.5，Expert 约为 5-6）。这表明模型已经具备了相当的竞争力，能够有效避免大量吃分，但在处理关键牌（如黑桃 Q）时仍有提升空间。

## ✨ 核心特性

- **🧠 Transformer 架构**: 专为卡牌游戏设计的神经网络，能够处理手牌、牌桌状态和历史出牌记录。
- **🎓 DAgger 预训练**: 相比传统的 Behavior Cloning，DAgger 让模型在自己的轨迹分布上学习 Expert 的策略，有效解决了分布偏移（Distribution Shift）问题。
- **⚔️ 强化学习 (PPO)**: 通过自我博弈（Self-Play）和 PPO 算法，让模型从胜负后果中学习，探索超越专家的策略。
- **🕵️ 增强型专家策略**: 内置的 Expert Policy 经过重构，具备确定性的决策逻辑，并包含高级的 **Shooting the Moon (射月)** 检测与执行逻辑。
- **🌐 Web 对战平台**: 提供一个基于 Web 的交互界面，允许人类玩家直接与 AI 进行对战。

## 📂 项目结构

| 文件 | 说明 |
| :--- | :--- |
| `train.py` | **训练主程序**。包含 DAgger 预训练和 PPO 训练循环，支持断点续传。 |
| `transformer.py` | **模型定义**。定义了 `HeartsTransformer` 网络结构。 |
| `strategies.py` | **策略库**。包含增强版 `ExpertPolicy`（专家）、`Random`（随机）等策略。 |
| `game.py` | **游戏引擎**。核心规则实现，支持训练模式和对战模式。 |
| `web_app.py` | **Web 服务器**。启动一个 Web 界面，允许用户与 AI 对战。 |
| `templates/` | **Web 前端**。包含游戏界面的 HTML/JS 代码。 |
| `showcase.py` | **展示脚本**。命令行版本的演示对战。 |

## 🚀 快速开始

### 1. 环境准备

确保安装了 Python 3.8+ 和 PyTorch。

```bash
pip install torch numpy flask
```

### 2. 启动 Web 对战

想亲自试一试 AI 的水平？运行 Web 服务器：

```bash
python web_app.py
```

然后打开浏览器访问 `http://localhost:5000` 即可开始游戏。

### 3. 重新训练

如果你想从头训练模型：

```bash
python train.py
```

程序会自动进行 Supervised Pretraining (DAgger) 和 PPO Training。

## 📈 策略演进

1.  **Stage 1 (Random Bots)**: 模型迅速学会规则，将平均分从 22 分降至 0 分（虐菜）。
2.  **Stage 2 (Expert Bots)**: 面对专家，模型初期表现下滑，但通过学习如何应对“逼牌”和“传牌陷阱”，分数逐渐稳定。
3.  **Stage 3 (Self-Play)**: 模型开始自我博弈，探索更复杂的纳什均衡策略。
