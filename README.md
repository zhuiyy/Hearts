# Hearts Research Platform

一个以**规则正确、信息隔离、可复现实验、代码可读**为目标的四人红心平台。

游戏规则、玩家策略、对局编排和统计评估通过明确的接口连接。
规则引擎负责单副与累计比赛，玩家通过观测选择动作，评估模块负责可复现的策略比较。

使用方式包括终端人工操作、完整回放、蒙特卡洛配对比较、block 级配对 t 检验和
Holm 多重比较校正。后续开发安排见 [任务与路线图](docs/roadmap.md)。

## 运行

需要 Python 3.11+。引擎、终端、回放和赛程使用 Python 标准库。

```bash
python -m hearts --help
python -m hearts play --seed 42 --record-dir artifacts/manual_001
python -m hearts play --seed 42 --match --record-dir artifacts/match_001
python -m hearts replay artifacts/manual_001/hand_0000.json
python -m hearts schedule --seed 42 --blocks 100 --split test --output artifacts/schedule_001.json
python -m unittest discover -s tests -v
```

`play` 在同一终端轮流操作四个座位。
编号输入三张换牌或一张出牌，`q` / Ctrl+C 退出。
终端清屏不能保证对共享电脑上的其他人保密；这里的 hot-seat 模式用于人工检查和本地同桌操作。
只有完整结束的一副才会保存全信息回放；文件已存在时拒绝覆盖。

需要统计推断时安装可选依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[stats]'
hearts analyze artifacts/comparison_001.json
```

`schedule` 生成配对赛程。`hearts.evaluation.compare` 接收策略工厂并执行比较，
比较结果通过 `Comparison.to_dict()` 转为字典，再用 `save_record` 保存。
`analyze` 读取比较结果并进行统计检验。

开发时可安装 `'.[stats,dev]'`，随后运行 `ruff check hearts tests` 和
`ruff format --check hearts tests` 检查代码规范与格式。

## 固定规则：`standard-hearts-v1`

红心没有唯一的全球统一规则，尤其首墩禁分、黑桃 Q 是否破门和射月计分存在变体。
本平台固定下面这一套常见的严格四人玩法，所有入口一致，不开放随意切换的规则选项。
规则参考：[Pagat 的 American Hearts](https://www.pagat.com/reverse/hearts.html)、
[Bicycle Hearts](https://bicyclecards.com/how-to-play/hearts)。两者在变体上并不完全一致，
发生差异时**以下条款是本平台的唯一判定标准**。

### 牌与座位

- 四人独立计分，无固定队伍；座位为 0、1、2、3，按此顺序顺时针行动。
- 标准 52 张牌，无大小王，每人 13 张，逐张顺时针发完。
- 同花色从小到大为 2、3、…、10、J、Q、K、A；无将牌。
- 终端用 C、D、S、H 表示梅花、方块、黑桃、红心，例如 `C2`、`SQ`、`HA`。

### 换牌

- 从第一副开始，依次循环：左传、右传、对传、不换。
- 本平台“左传”定义为传给下一顺时针座位，即 `(seat + 1) % 4`；
  右传为 `(seat + 3) % 4`，对传为 `(seat + 2) % 4`。
- 需要换牌时，每人必须从自己的 13 张牌中选三张不同的牌。梅花 2 也允许传出。
- 四人的选择全部提交后统一交换。在选择完自己传出的牌之前，不能看到收到的牌。
- 提交期间不提前删除手牌，不提前传递任何牌；四人交换完成后仍各持 13 张。

### 出牌

1. 换牌后持有梅花 2 的玩家首先行动，第一张必须出梅花 2。
2. 其余玩家顺时针行动。有首出花色时必须跟牌，无该花色时才可以垫其他花色。
3. **第一墩禁止垫红心或黑桃 Q**；唯一例外是无法跟梅花且手中全部为计分牌。
   该例外保证任何合法发牌都能继续，不能为了“禁分”造成无合法动作。
4. 第一墩之后，缺门时可以垫任意牌，包括红心或黑桃 Q；不需要等红心破门。
5. 红心尚未破门时不能领红心，除非手里只剩红心。该强制例外也会使红心破门。
6. 只有实际打出红心才算破门；**打出黑桃 Q 不会破门**。
7. 每墩由首出花色中点数最大的牌获胜，获胜玩家收走整墩并领下一墩。
8. 黑桃 Q 在合法跟牌、垫牌或非首墩领牌时均可打出，没有“能甩必须甩”的限制。
9. 一副恰好 13 墩、52 次出牌。出完后统一做本副最终计分。

### 计分与比赛结束

- 每张红心 1 分，黑桃 Q 为 13 分，其他牌 0 分；罚分越少越好。
- 正常一副原始罚分合计 26 分。
- 若一人吃到全部 13 张红心和黑桃 Q，即射月：该玩家本副记 0 分，其他三人各记 26 分。
  不提供“自己减 26 分”的可选项，也没有射日、方块 J 减分或恰好 100 分回退等额外规则。
- 累计比赛只在一副完全结束并结算后检查：有人累计达到或超过 100 分即结束。
- 此时累计最低分者获胜；最低分并列时共享胜利，不任意选择一人，也不额外加赛。
- 引擎保留射月前的 `raw_scores` 和射月后的 `scores`，避免把两种量混在一起。

## 接口边界

```python
class Agent(Protocol):
    def act(self, observation: Observation) -> PassAction | PlayAction:
        ...
```

- 玩家只收到自己的手牌、自己的传出/接收记录、公开历史和当前合法选项。
- 公开状态没有对手手牌、发牌种子、其他玩家换牌内容或真实隐藏状态标签。
- `AgentSpec(name, create)` 的工厂为每个座位、每一副创建独立实例。
- `AgentContext.seed` 只属于该玩家；策略自行初始化局部 RNG，不使用全局随机状态。
- 所有游戏动作最终只经过 `HeartsHand.submit`；非法动作报错且不改变游戏。
- 公共数据为不可变值，不能通过修改观测来修改引擎。
- 全信息初始牌局、动作记录及结束后的回放属于可信编排层，不能传给策略。
- 集中式 critic 等训练组件需要的全局状态，应通过独立的特权通道提供。

## 比较与随机性

- 比较固定一个焦点策略与三个对手：将焦点分别换成 baseline / candidate。
- 每个独立牌局块复用同一副初始牌，交叉四种换牌方向、四个座位，共 16 个配对试验。
- 每个试验两组均运行，所以 `B` 个 block 共需要 `32B` 副完整模拟。
- 两组共享初始牌、阵容相对位置、换牌方向及玩家初始 RNG 流。
- 发牌与策略随机流分离；增加策略采样次数不会改变后续发牌。
- train / validation / test 使用不同的派生路径。技术隔离不替代禁止用最终测试集调参的实验纪律。
- 每个 block 先平均 16 个“基线罚分 − 候选罚分”，正值表示候选更好。
- t 检验的样本量是独立牌局块数，不是 16 倍的轮换次数；默认双侧检验，报告效应与置信区间。
- 多项预先指定的比较可用 `holm_adjust` 校正 p 值；它不校正反复查看后停止或测试集选优。
- 比较以固定策略的单副罚分为目标；多个训练种子的变异需在独立的统计层次分析。

详细公式、独立性假设、数据结构选择和代码阅读顺序见 [架构说明](docs/architecture.md)。

## 目录与审查顺序

```text
hearts/
  cards.py         牌与稳定编号
  types.py         公开状态、私有观测、动作、结果
  seeds.py         命名随机流
  deal.py          显式初始牌局及校验
  rules.py         纯规则函数
  engine.py        唯一状态机
  agents.py        策略协议、创建工厂与玩家上下文
  arena.py         单副和累计比赛编排
  replay.py        完整记录、保存与重放验证
  evaluation.py    蒙特卡洛配对比较
  statistics.py    block 级 t 检验、Holm 校正
  terminal.py      人工输入与文本展示
  cli.py           命令入口
tests/            测试专用出牌器和自动验证
docs/             架构说明与后续任务
```

建议按上述顺序审查。模块设计注重逻辑透明和接口稳定，性能优化以实际测量为依据。
