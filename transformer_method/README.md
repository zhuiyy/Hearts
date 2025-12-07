# Hearts AI (Transformer + PPO + DAgger)

这是一个基于深度强化学习（Deep RL）的红心大战（Hearts）AI 项目。项目采用 **Transformer** 架构作为策略网络，结合了 **DAgger (Dataset Aggregation)** 算法进行监督预训练，并通过 **PPO (Proximal Policy Optimization)** 算法进行强化学习微调。

经过精心设计的课程学习（Curriculum Learning），该 AI 已经从最初的随机出牌进化为能够与专家级对手抗衡的高手。

## ✨ 核心特性

- **🧠 Transformer 架构**: 专为卡牌游戏设计的神经网络，能够处理手牌、牌桌状态和历史出牌记录。
- **🚀 多 GPU 并行训练**: 支持 `DataParallel`，可利用多张显卡加速训练过程。
- **🎓 DAgger 预训练**: 相比传统的 Behavior Cloning，DAgger 让模型在自己的轨迹分布上学习 Expert 的策略，有效解决了分布偏移（Distribution Shift）问题。
- **⚔️ 强化学习 (PPO)**: 通过自我博弈（Self-Play）和 PPO 算法，让模型从胜负后果中学习，探索超越专家的策略。
- **🕵️ 增强型专家策略**: 内置的 Expert Policy 经过重构，具备确定性的决策逻辑，并包含高级的 **Shooting the Moon (射月)** 检测与执行逻辑。
- **🌐 Web 对战平台**: 提供一个基于 Web 的交互界面，允许人类玩家直接与 AI 进行对战。

## 📂 项目结构

项目经过模块化重构，结构更加清晰：

| 文件 | 说明 |
| :--- | :--- |
| `config.py` | **全局配置**。集中管理超参数（LR, Batch Size）、文件路径和训练设置。 |
| `pretrain.py` | **预训练脚本**。执行监督学习（DAgger），生成 `hearts_model_pretrained.pth`。 |
| `train.py` | **PPO 训练主程序**。加载预训练模型（可选），进行强化学习微调。 |
| `agent.py` | **Agent 定义**。包含 `AIPlayer` 类和 `OpponentPool`（对手池）。 |
| `transformer.py` | **模型定义**。定义了 `HeartsTransformer` 网络结构。 |
| `strategies.py` | **策略库**。包含增强版 `ExpertPolicy`（专家）、`Random`（随机）等策略。 |
| `game.py` | **游戏引擎**。核心规则实现，支持训练模式和对战模式。 |
| `web_app.py` | **Web 服务器**。启动一个 Web 界面，允许用户与 AI 对战。 |
| `showcase.py` | **展示脚本**。命令行版本的演示对战。 |
| `saved_models/` | **模型存储**。所有的 `.pth` 模型文件都会自动保存到此目录。 |

## 🚀 快速开始

### 1. 环境准备

确保安装了 Python 3.8+ 和 PyTorch。

```bash
pip install torch numpy flask
```

### 2. 训练流程

推荐的训练工作流如下：

**第一步：监督预训练 (Supervised Pretraining)**
让 AI 快速学会基础规则和专家策略。
```bash
python pretrain.py
```
*   支持多 GPU 选择。
*   训练完成后，模型将保存为 `saved_models/hearts_model_pretrained.pth`。

**第二步：强化学习 (PPO Training)**
在预训练的基础上，通过自我博弈进一步提升。
```bash
python train.py
```
*   程序会自动检测是否存在预训练模型，并询问是否加载。
*   支持多 GPU 并行训练。
*   最终模型将保存为 `saved_models/hearts_model.pth`。

### 3. 启动 Web 对战

想亲自试一试 AI 的水平？运行 Web 服务器：

```bash
python web_app.py
```

然后打开浏览器访问 `http://localhost:5000` 即可开始游戏。

### 4. 命令行演示

观看 AI 与 AI 之间的对战演示：

```bash
python showcase.py
```

## 📈 训练策略演进

1.  **Stage 1 (Random Bots)**: 模型迅速学会规则，将平均分从 22 分降至 0 分（虐菜）。
2.  **Stage 2 (Expert Bots)**: 面对专家，模型初期表现下滑，但通过学习如何应对“逼牌”和“传牌陷阱”，分数逐渐稳定。
3.  **Stage 3 (Self-Play)**: 模型开始自我博弈，探索更复杂的纳什均衡策略。

## 🛠️ 配置说明

所有超参数均在 `config.py` 中定义，你可以轻松修改：
*   `LEARNING_RATE`: 学习率
*   `BATCH_SIZE`: 批次大小
*   `HIDDEN_DIM`: 模型隐藏层维度
*   `THROTTLE_TIME`: 训练时的 GPU 降温/节流时间

## 📝 自动提交脚本

项目包含一个便捷的自动提交脚本 `git_auto_push.bat` (Windows)，双击即可自动执行 `git add`, `commit` (带时间戳) 和 `push`。
