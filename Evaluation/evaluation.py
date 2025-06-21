import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import math
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
import json
from cal_indicator import (
    create_nx_graph_from_data,
    define_risk_patterns_from_json,
    calculate_ddrr,
    calculate_radi_sliding_window,
    calculate_crra,
    calculate_gcri,
    classify_risk_level,
)


def get_normalized_text_embedding(text, model_path="bert-base-uncased", device="cpu"):
    max_length = 256

    # 加载 tokenizer 和模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    # 分词并转换为 token IDs
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    ).to(device)

    # 计算 embedding
    with torch.no_grad():
        outputs = model(**inputs)

    # 取所有 token 的平均作为文本 embedding
    embeddings = outputs.last_hidden_state.mean(dim=1)  # [1, d]

    # L2 归一化（使其严格满足 ||e||_2 = 1）
    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)  # [1, d]

    return normalized_embeddings.squeeze(0)  # 返回 [d] 的张量


def add_node(timestamp, t, r, string, model_path="bert-base-uncased"):
    global vectices
    e = get_normalized_text_embedding(
        string,
        model_path=model_path,
        device="cuda:1",
    )
    v = torch.cat([torch.tensor([timestamp, t, r]), e.cpu()], dim=0)
    vectices.append(v)


def add_edge(user_idx, rsp_idx, s_ij=0):
    global edge_idx, edge_weight, vectices
    # 联通边
    edge_idx[0].append(user_idx)
    edge_idx[1].append(rsp_idx)

    # 计算边的权重
    edge_set = []
    sim = cosine_similarity(
        vectices[user_idx][3:].unsqueeze(0).cpu().numpy().reshape(1, -1),
        vectices[rsp_idx][3:].unsqueeze(0).cpu().numpy().reshape(1, -1),
    )[0][0]
    delta_t = abs(
        vectices[user_idx][1].to(torch.int) - vectices[rsp_idx][1].to(torch.int)
    )
    lambda_value = math.log(2 / 3)  # λ = ln(2/3)
    w_ij = sim * torch.exp(-lambda_value * delta_t)
    edge_set.append(w_ij)
    edge_set.append(delta_t)
    edge_set.append(torch.tensor(s_ij))
    edge_weight.append(edge_set)


def rpg_algorithm_add_node(
    cur_conv_num, agent_num, data, model_path="bert-base-uncased"
):
    global vectices, edge_num, role_list
    user_idx = cur_conv_num * (agent_num + 1)
    rsp_idx = user_idx + 1
    add_node(
        # timestamp=cur_conv_num,
        timestamp=0,
        t=0,
        # r=0标识用户id
        r=0,
        # 输入user的提问
        string=data["rounds"][cur_conv_num][f"question{cur_conv_num + 1}"],
        model_path=model_path,
    )
    role_list.append(0)
    for i in range(agent_num):
        add_node(
            # timestamp=cur_conv_num,
            timestamp=i + 1,
            t=1,
            r=data["rounds"][cur_conv_num]["role_id"],
            # 输入agent的回答
            string=data["rounds"][cur_conv_num][f"answer{i + 1}"],
            model_path=model_path,
        )
        role_list.append(data["rounds"][cur_conv_num]["role_id"])
        add_edge(
            user_idx=user_idx,
            rsp_idx=rsp_idx,
            s_ij=0,
        )
        rsp_idx += 1


class RiskGAT(GATConv):
    def __init__(
        self,
        in_channels,
        out_channels,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0,
        add_self_loops=False,
        bias=True,
        **kwargs,
    ):
        super(RiskGAT, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            concat=concat,
            negative_slope=negative_slope,
            dropout=dropout,
            add_self_loops=add_self_loops,
            bias=bias,
            **kwargs,
        )

    def forward(self, data, return_attention_weights=True):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 调用父类的forward方法
        out, (edge_index, alpha) = super().forward(
            x,
            edge_index,
            return_attention_weights=return_attention_weights,
            # edge_attr=edge_attr[:, 0].unsqueeze(1) if edge_attr is not None else None,
            edge_attr=edge_attr if edge_attr is not None else None,
        )

        # 返回节点特征和注意力分数
        return out, alpha


lines = []
with open("./example/test.jsonl", "r") as f:
    lines = f.readlines()
data = json.loads(lines[1])
# ------------------- 数据转换为图 -------------------
conv_num = data["round_num"]
agent_num = data["agent_num"]
vectices = []
edge_idx = [[], []]
edge_weight = []
role_list = []
V = None
edge_num = 0
threshold = 0.6
model_path = r"/mnt/data/MASCSBench/model/bert-base-uncased"
for i in range(conv_num):
    rpg_algorithm_add_node(i, agent_num, data, model_path=model_path)
V = torch.stack(vectices)
# 添加相似度高的边
for i in range(V.shape[0]):
    for j in range(V.shape[0]):
        if abs(j - i) < agent_num:
            sim = cosine_similarity(
                vectices[i][3:].unsqueeze(0).cpu().numpy().reshape(1, -1),
                vectices[j][3:].unsqueeze(0).cpu().numpy().reshape(1, -1),
            )[0][0]
            if sim > threshold:
                add_edge(
                    user_idx=i,
                    rsp_idx=j,
                    s_ij=0,
                )
# 更新角色的权重
role_list = list(set(role_list))
updated_list = []
for p in role_list:
    for q in role_list:
        if (p, q) not in updated_list and (q, p) not in updated_list:
            updated_list.append((p, q))
        else:
            continue
        # 计算分母
        p_num = 0
        q_num = 0
        for k in range(V.shape[0]):
            if vectices[k][2] == p:
                p_num += 1
            if vectices[k][2] == q:
                q_num += 1
        all_weight = 0
        update_role = []
        for k in range(len(edge_idx[0])):
            if (edge_idx[0][k] == p and edge_idx[1][k] == q) or (
                edge_idx[0][k] == q and edge_idx[1][k] == p
            ):
                all_weight += edge_weight[k][0]
                update_role.append(k)
        all_weight = all_weight / (p_num * q_num)
        for k in update_role:
            edge_weight[k][0] = all_weight
            edge_weight[k][2] = torch.tensor(1)

edge_index = torch.tensor(edge_idx, dtype=torch.long)
edge_attr = torch.stack([torch.stack(row) for row in edge_weight], dim=0).float()
# ------------------- 模型初始化与推理 -------------------
model_data = Data(x=V, edge_index=edge_index, edge_attr=edge_attr)  # 自定义时间差属性
model = RiskGAT(in_channels=771, out_channels=1)
_, attention_weights = model(model_data)

# 计算自适应阈值θ_r = μ + 2σ (论文公式)
risk_scores = torch.zeros(conv_num * (agent_num + 1))
for i in range(conv_num * (agent_num + 1)):
    for j in range(len(attention_weights)):
        if edge_index[0][j] == i:
            risk_scores[i] += attention_weights[j][0] * edge_attr[j][0]
    risk_scores[i] = risk_scores[i] / V.shape[0]
mu = torch.mean(risk_scores)
sigma = torch.std(risk_scores)
theta_r = mu + 2 * sigma

print(f"节点风险评分:\n{risk_scores.detach().numpy()}")
print(f"自适应阈值θ_r = {theta_r.item():.4f}")


# # 计算自适应阈值θ_r = μ + 2σ (论文公式)
# risk_scores = torch.zeros(conv_num * (agent_num + 1))

# # 直接使用attention_weights作为alpha张量
# alpha = attention_weights.squeeze(1)  # 将形状从 [num_edges, 1] 转换为 [num_edges]

# # 计算出度和入度
# out_degree = torch.zeros_like(risk_scores)
# in_degree = torch.zeros_like(risk_scores)
# for j in range(edge_index.shape[1]):
#     src = edge_index[0][j].item()
#     dst = edge_index[1][j].item()
#     out_degree[src] += 1
#     in_degree[dst] += 1

# # 计算风险贡献（双向传播）
# for j in range(edge_index.shape[1]):
#     src = edge_index[0][j].item()
#     dst = edge_index[1][j].item()

#     # 获取当前边的属性
#     edge_w = edge_attr[j][0].item()  # 边权重
#     attn_w = alpha[j].item()  # 注意力权重

#     # 更新起点和终点的风险评分
#     risk_scores[src] += attn_w * edge_w  # 起点贡献
#     risk_scores[dst] += attn_w * edge_w  # 终点贡献

# # 归一化处理（按总连接数）
# total_degree = out_degree + in_degree
# total_degree[total_degree == 0] = 1  # 避免除零
# risk_scores = risk_scores / total_degree

# # 计算统计量
# mu = torch.mean(risk_scores)
# sigma = torch.std(risk_scores)
# theta_r = mu + 2 * sigma

# print(f"节点风险评分:\n{risk_scores.detach().numpy().round(4)}")
# print(f"自适应阈值θ_r = {theta_r.item():.4f}")
print(f"注意力权重:\n{attention_weights}")

# ------------------- 在已有代码末尾添加DDRR计算 -------------------
# 转换图为networkx格式
dialogue_nx_graph = create_nx_graph_from_data(V, edge_index, edge_attr)

# 加载风险模式（示例）
# risk_patterns = define_risk_patterns()
risk_patterns = define_risk_patterns_from_json("risk_patterns.json")

# 计算DDRR
ddrr_score = calculate_ddrr(dialogue_nx_graph, risk_patterns)
print(f"\nDDRR Score: {ddrr_score:.4f}")

# 计算RADI
# 生成时间戳（假设每个节点的时间戳存储在V的第一列）
timestamps = V[:, 0].numpy()
radi_score = calculate_radi_sliding_window(
    risk_scores, timestamps, theta_r=mu + 3 * sigma, conv_num=conv_num
)
print(f"\nRADI Score: {radi_score:.4f}")

# 计算CRRA
lambda_decay = math.log(2) / 3  # 论文中λ=ln2/3
crra_score = calculate_crra(V, edge_index, edge_attr, role_list, lambda_decay)
print(f"CRRA Score: {crra_score:.4f}")

# 计算GCRI
print(f"\n{'='*50}")
print(f"GCRI (Global Conversation Risk Index) Calculation")
print(f"{'='*50}")

# 提取对话文本
conversation_texts = []
for i in range(conv_num):
    # 用户问题
    user_question = data["rounds"][i][f"question{i + 1}"]
    conversation_texts.append(user_question)

    # 智能体回答（选择第一个回答作为代表）
    agent_answer = data["rounds"][i]["answer1"]
    conversation_texts.append(agent_answer)

print(f"Extracted {len(conversation_texts)} conversation segments")

# 提取时间戳（对话轮次）
timestamps = []
for i in range(conv_num):
    timestamps.extend([i, i])  # 每轮对话包含用户和智能体两个时间点

# 使用与对话文本长度匹配的风险评分
# 如果风险评分长度不匹配，进行调整
if len(risk_scores) != len(conversation_texts):
    # 重新计算每个对话段的风险评分
    node_risk_scores = []
    text_idx = 0
    for i in range(conv_num):
        # 用户节点风险评分
        user_node_idx = i * (agent_num + 1)
        if user_node_idx < len(risk_scores):
            node_risk_scores.append(risk_scores[user_node_idx].item())
        else:
            node_risk_scores.append(0.0)

        # 智能体节点风险评分（取第一个智能体）
        agent_node_idx = user_node_idx + 1
        if agent_node_idx < len(risk_scores):
            node_risk_scores.append(risk_scores[agent_node_idx].item())
        else:
            node_risk_scores.append(0.0)

        text_idx += 2
else:
    node_risk_scores = (
        risk_scores.detach().numpy()
        if isinstance(risk_scores, torch.Tensor)
        else risk_scores
    )

# 确保长度匹配
min_len = min(len(node_risk_scores), len(conversation_texts), len(timestamps))
node_risk_scores = node_risk_scores[:min_len]
conversation_texts = conversation_texts[:min_len]
timestamps = timestamps[:min_len]

print(f"Using {len(node_risk_scores)} risk scores for GCRI calculation")

# 计算GCRI
lambda_decay_gcri = math.log(2) / 3.47  # 论文中的λ值
gcri_result = calculate_gcri(
    risk_scores=node_risk_scores,
    timestamps=timestamps,
    conversation_texts=conversation_texts,
    alpha=0.5,  # TRS权重
    beta=0.5,  # CRD权重
    lambda_decay=lambda_decay_gcri,
)

# 风险等级分类
risk_classification = classify_risk_level(gcri_result["GCRI"])

# 输出结果
print(f"\nGCRI Analysis Results:")
print(f"TRS (Temporal Risk Score): {gcri_result['TRS']:.4f}")
print(f"CRD (Contextual Risk Density): {gcri_result['CRD']:.4f}")
print(f"GCRI (Global Conversation Risk Index): {gcri_result['GCRI']:.4f}")
print(f"Risk Level: {risk_classification['level']}")
print(f"Recommended Action: {risk_classification['action']}")
