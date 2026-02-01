# DDRR-GNN v1（方案A：双塔图嵌入 + ANN检索）设计文档

> 目标：用**图嵌入检索**替代“WordNet 语义匹配 + 子图同构”，构建一个更快、更鲁棒、适用于多轮多 agent 对话的 DDRR 计算方式。  
> 约束：本方案为**方案A（最轻量）**，不引入交互式精排；实现时可先做到“可跑 + 可调参 + 可回溯”，后续再升级方案B/C。

---

## 1. 背景与问题

### 1.1 现有 DDRR 的瓶颈
现有 DDRR 以风险模式库 \( \{P_k\}_{k=1..C} \) 为参照，通过：
- WordNet（Wu-Palmer 等）做节点/关键词语义对齐
- 子图同构匹配（pattern ⊑ dialog graph）判断命中

在多轮对话、节点多、agent 多、模式库大时，整体复杂度高，难以在线计算与快速迭代。

### 1.2 新方案原则
- 将硬匹配 \(I(P_k \sqsubseteq G)\) 替换为**可学习/可计算**的相似度 \(s_k\)
- 通过 **双塔编码 + 向量检索**，把“全库遍历”变成“Top-M 候选检索”
- 保留 DDRR 的指标直觉：命中越多（越高权重）的风险模式，DDRR 越高

---

## 2. 输入输出与使用方式

### 2.1 输入：多轮对话 jsonl（参考格式）
每一行是一条消息（或一个 turn 的一段），至少包含：
- `conversation_id`
- `turn_id`（轮次）
- `speaker/agent`（角色或 agent 标识）
- `text`（内容）
- 可选：`timestamp`、`meta`、`tool_call`、`system_prompt` 等

> 实现时建议统一做一个 **Message Schema**（见 3.1）。

### 2.2 输出：DDRR 分数（可选输出解释）
最小输出：
- `ddrr_soft`（0~1）
- `ddrr_hard`（0/1 版本或 0~1 但阈值化）
- `matched_patterns_top`（Top-N 命中的模式及分数，便于审计）

---

## 3. 数据结构定义

### 3.1 Message Schema（对话消息）
建议内部标准化后结构如下：

- `cid`: string  
- `turn`: int  
- `agent`: string（如 user/assistant/agentA/agentB）  
- `role`: enum（user/assistant/system/tool/agent）  
- `text`: string  
- `time`: float/int（可选）  
- `extra`: dict（可选）

### 3.2 风险模式库 Pattern Library
每个模式 \(P_k\) 最少需要：

- `pattern_id`: string  
- `name`: string（简短描述）  
- `description`: string（可更长）  
- `weight` = \(\omega_k\): float（风险权重）  
- `risk_tags`: list（类别/维度，例如 jailbreak、malware、self-harm 等）  
- `pattern_text_units`: list[string]（用于编码的文本单元集合，见 4.2）  
- 可选：`version`、`severity`、`source`、`examples`

> 方案A不强制要求模式是“图结构”，但为了与后续方案B/C兼容，建议保留可扩展字段（例如 slots/edges）。

### 3.3 对话图 Graph（RPG 的轻量版）
方案A核心不依赖显式子图同构，但仍建议构建一个轻量对话图 \(G\) 以便：
- 形成“图嵌入”（Graph Embedding）
- 支持多轮窗口化与在线更新

节点（Node）：
- 一条消息一节点（或对一条消息分句为多节点，取决于你们现有粒度）
- 节点属性：`agent_id`, `turn`, `text_embedding`, `role_embedding`, `time_embedding`

边（Edge）：
- `adjacent_turn`：同一 agent 相邻轮次连接
- `semantic_sim`：语义相似度大于阈值的连接（可稀疏 top-k）
- `cross_agent`：跨 agent 交互边（例如同轮 user->assistant 或引用关系）

> 方案A可先“弱图化”：只保留相邻轮次边 + 同轮交互边，尽量简单。

---

## 4. 表示学习与编码方式（双塔）

### 4.1 总体结构
- **Dialog Encoder（塔G）**：输入对话（或对话图/窗口图）→ 输出向量 \(z_G \in \mathbb{R}^d\)
- **Pattern Encoder（塔P）**：输入模式文本（或模式图）→ 输出向量 \(z_{P_k} \in \mathbb{R}^d\)

相似度：
\[
s_k = \cos(z_G, z_{P_k}) \quad \text{或} \quad s_k = \frac{z_G \cdot z_{P_k}}{\|z_G\|\|z_{P_k}\|}
\]

### 4.2 Pattern Encoder 输入组织（关键点）
为了让模式库“可检索 + 可维护”，建议为每个模式构建 `pattern_text_units`，例如：
- 单句摘要：`"how to bypass safety policy"`（英文/中文皆可）
- 风险标签拼接：`"[malware] [evasion] [instruction]" + description`
- 典型触发表达（可选）：若你们已有例句，可加入 3~10 条 canonical prompts

最终模式向量 \(z_{P_k}\) 可以是：
- 对 units 的 embedding 做 mean/max pooling
- 或者 units 拼接成一段文本后编码（更简单）

> 离线预计算所有 \(z_{P_k}\)，并建立 ANN 索引。

### 4.3 Dialog Encoder 输入组织（在线友好）
对话侧建议同时支持两种输入：
1) **全量对话编码**：适合离线评测
2) **滑动窗口编码**：适合在线监测与长对话

窗口定义：
- 最近 \(W\) 轮（例如 W=3/5/8）
- 或最近 \(T\) 条消息（例如最近 20 条）
- 可增加 “历史摘要节点”（可选）：用一句摘要代表早期内容

输出向量 \(z_G\) 的构成建议：
- 简版：对窗口内消息 embedding 做 pooling + 角色/时间特征融合
- 图版：对窗口图做一次轻量 GNN 聚合再 pooling（仍属于方案A，只要最终输出一个向量即可）

---

## 5. 检索与命中判断

### 5.1 ANN 检索（候选生成）
离线：为所有模式向量 \(z_{P_k}\) 建立 ANN 索引。  
在线：对每个对话窗口得到 \(z_G\)，检索 Top-M 候选：

- 候选集合：\(\mathcal{C}_M = \text{TopM}(\cos(z_G, z_{P_k}))\)
- 只对 \(\mathcal{C}_M\) 计算/更新分数（避免全库遍历）

推荐默认：
- \(M=50\)（模式库较大时 50~200 可调）
- 输出 Top-N 命中解释（例如 N=10）

### 5.2 分数校准：从相似度到“命中概率”
定义一个可解释、可调的映射函数 \(f(\cdot)\) 把 \(s_k\) 映射到 \([0,1]\)：

**选项A：Sigmoid**
\[
p_k = \sigma(\alpha(s_k - \beta))
\]
- \(\alpha\)：陡峭度（默认 10~30）
- \(\beta\)：中心阈（可按验证集调）

**选项B：分段线性**
- \(s_k < \tau_{low} \Rightarrow p_k=0\)
- \(s_k > \tau_{high} \Rightarrow p_k=1\)
- 中间线性插值

> 建议先用分段线性，直观易调；后续再换 sigmoid 并做温度标定。

### 5.3 多窗口融合（多轮对话）
对话进行过程中会产生多个窗口 \(G^{(t)}\)（t 为轮次），对同一模式的匹配分数可融合：

- **最大值融合**（强调“出现过就算”）：
\[
p_k = \max_t p_k^{(t)}
\]

- **时间衰减融合**（强调“越新越重要”）：
\[
p_k = \max_t \left(\gamma^{(T-t)} \cdot p_k^{(t)}\right), \quad \gamma \in (0,1)
\]

> 推荐默认：max 融合 + 可选时间衰减（\(\gamma=0.9\sim0.99\)）。

---

## 6. 新 DDRR 定义（方案A）

### 6.1 软 DDRR（推荐主指标）
令模式权重为 \(\omega_k\)，真实风险集合为 \(R\)（评测时可用）。  
定义软命中为 \(p_k \in [0,1]\)。

\[
DDRR_{soft}=\frac{\sum_{k=1}^{C}\omega_k\cdot p_k}{\sum_{k=1}^{C}\omega_k\cdot I(P_k\in R)}
\]

解释：
- 分子：对所有模式的“命中程度”加权求和（实际可只对候选近似）
- 分母：真实风险模式的权重和（评测用；线上若无 R，可用全库权重和或另定义）

### 6.2 硬 DDRR（用于可比性/阈值告警）
定义硬命中：
\[
\hat{I}_k = I(p_k > \delta_k)
\]
则：
\[
DDRR_{hard}=\frac{\sum_{k=1}^{C}\omega_k\cdot \hat{I}_k}{\sum_{k=1}^{C}\omega_k\cdot I(P_k\in R)}
\]

> \(\delta_k\) 可统一取常数（例如 0.5），或按类别/模式单独校准。

### 6.3 线上无 R 的 DDRR-online（可选）
线上往往没有真实 R。可定义一个“风险覆盖率”类指标：

- 方案1：用全库权重归一（得到 0~1 风险强度）
\[
DDRR_{online}=\frac{\sum_{k=1}^{C}\omega_k\cdot p_k}{\sum_{k=1}^{C}\omega_k}
\]

- 方案2：只对风险类别子集归一（例如只看 malware 类）

---

## 7. 训练与标注（可选，但推荐）

> 方案A可以“无训练”先跑：直接用通用 embedding 生成 \(z_G, z_{P_k}\)。  
> 但为了准确率与领域适配，建议加入轻量训练或对比学习。

### 7.1 训练数据构造
- 正样本：对话 \(G\) 与其真实风险模式 \(P^+\in R\)
- 负样本：随机采样模式 \(P^-\notin R\)，以及 hard negatives（相似但不属于该风险）

### 7.2 目标函数（示例）
- 对比学习/排序损失（让 \(s(G,P^+)\) 大于 \(s(G,P^-)\)）
- 或二分类（匹配/不匹配）

### 7.3 校准与阈值
- 在验证集上为每类/每模式拟合 \(\tau_{low}, \tau_{high}\) 或 sigmoid 的 \(\alpha,\beta\)
- 输出 PR 曲线选择阈值，针对“漏报/误报”偏好调参

---

## 8. 可解释性与审计输出（方案A可做的解释）

尽管方案A不做精排对齐，仍可输出“弱解释”：

对每个命中模式 \(P_k\)（Top-N）输出：
- `pattern_id`, `name`, `risk_tags`, `weight`
- `score`: \(s_k\)
- `p`: \(p_k\)
- `window_id / turn_range`: 哪个窗口产生最大匹配
- `evidence_messages`: 该窗口内 Top-K 相似的消息（用消息 embedding 与模式 embedding 的相似度挑选）

> 这能满足基础审计：知道“为什么这个模式命中”。

---

## 9. 性能与复杂度预估（方向性）

- 原方案：全库同构匹配 + WordNet 对齐，随对话长度、模式库规模快速膨胀
- 方案A：
  - 离线：对 \(C\) 个模式预计算向量 + 建索引
  - 在线：每个窗口只需一次编码（或少量聚合）+ ANN Top-M 检索
  - 复杂度约为：\(O(\text{encode}(G^{(t)})) + O(\log C)\)（检索近似）

---

## 10. 参数建议（初始默认）

- 向量维度：d = 768/1024（随 embedding 模型）
- 窗口大小：W = 5 轮（或最近 20 条消息）
- ANN Top-M：M = 50（大库可 100~200）
- 分段线性阈值：\(\tau_{low}=0.25\), \(\tau_{high}=0.45\)（仅示例，需验证集调）
- 多窗口融合：max 融合；如需更“在线敏感”，加时间衰减 \(\gamma=0.95\)

---

## 11. 与后续方案B/C的兼容预留

为避免未来推倒重来，建议现在就预留：
- Pattern 的结构字段（slots/edges）即使方案A不用
- 输出中保留 `candidate_patterns`（Top-M）与 `evidence_messages`
- Dialog Graph 构建保留节点 id、turn、agent 等字段，方便方案B做跨图对齐

---

## 12. 验收标准（建议）

### 12.1 正确性
- 在有真实 R 的评测集上：
  - DDRR_soft 与原 DDRR 的相关性（Spearman/Pearson）
  - 基于阈值告警的 Precision/Recall/F1
- 能覆盖多 agent、多轮场景

### 12.2 效率
- 单对话（长度 L，窗口数 T）推理时间随 C 增长近似对数/亚线性
- 支持在线更新（每新增一轮只更新一次窗口编码与检索）

### 12.3 可解释性
- Top-N 命中模式可回溯到具体窗口与具体消息片段

---

## 13. 你可以直接在 Codex 里 vibe coding 的实现清单（不含代码）

按模块拆分：
1) `schema/`：Message / Pattern / Output 定义
2) `pattern_index/`：模式向量离线生成、存储、ANN 索引
3) `dialog_encoder/`：窗口化、编码、（可选）轻量图聚合
4) `retrieval/`：Top-M 检索与打分、分数到概率映射
5) `ddrr/`：DDRR_soft / DDRR_hard / DDRR_online 计算
6) `explain/`：证据窗口、证据消息提取
7) `eval/`：与原 DDRR 或标签集对齐评测、阈值校准

---

## 14. 备注：方案A的边界与升级路径

- 方案A在“严格结构依赖”的风险模式上可能不如子图同构敏感  
  但可通过：
  - 窗口化 + 历史摘要
  - Pattern text_units 更精细（加入槽位描述与触发表达）
  - 轻量图聚合（仍输出单向量）  
  获得显著提升。

- 若需要更强的“模式槽位 ↔ 对话节点”对齐解释与精度：
  - 升级到方案B：检索（A） + 交互式精排（Cross-Attention Graph Matching）

---
