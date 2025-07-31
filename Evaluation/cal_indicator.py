import os
import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
import numpy as np
import torch
import math
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import json
import nltk
from similarity_comparison import *
from itertools import combinations
from fast_similarity_matcher import *


# ======================== DDRR计算 ========================
def create_nx_graph_from_data(V, edge_index, edge_attr, all_dialogue):
    """将PyG图数据转换为networkx图"""
    G = nx.DiGraph()

    # 添加节点
    for i in range(V.shape[0]):
        node_attrs = {
            "timestamp": int(V[i][0]),
            "type": int(V[i][1]),
            "role": int(V[i][2]),
            "embedding": V[i][3:],
            "text": all_dialogue[i] if i < len(all_dialogue) else "",
        }
        G.add_node(i, **node_attrs)

    # 添加边
    for idx, (src, dst) in enumerate(edge_index.T):
        edge_attrs = {
            "weight": float(edge_attr[idx][0]),
            "delta_t": int(edge_attr[idx][1]),
            "s_ij": int(edge_attr[idx][2]),
        }
        G.add_edge(int(src), int(dst), **edge_attrs)

    return G


def define_risk_patterns_from_json(json_path="risk_patterns.json", nltk_data_path="nltk_data", alpha=0.7):
    """从 risk_patterns.json 构造结构化风险模式图列表，带有严重度估算"""

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.environ['NLTK_DATA'] = nltk_data_path
    nltk.data.path.insert(0, nltk_data_path)
    patterns = []
    max_severity = 0.0

    node_type_map = {
        "material_keywords": 10,
        "process_keywords": 11,
        "dangerous_items": 12,
        "finance_keywords": 20,
        "urgency_keywords": 21,
        "sensitive_info": 22,
        "identity_fields": 30,
        "context_keywords": 31,
        "medicine_keywords": 40,
        "abuse_keywords": 41,
        "tech_keywords": 50,
        "attack_keywords": 51,
        "statistics_keywords": 60,
        "discrimination_keywords": 61,
        "authority_keywords": 70,
        "advice_keywords": 71,
        "ip_keywords": 80,
        "bypass_keywords": 81,
        "psychology_keywords": 90,
        "manipulation_keywords": 91,
        "neutral_terms": 100,
        "extreme_content": 101,
    }

    for category, pattern_list in data.items():
        for item in pattern_list:
            G = nx.DiGraph()
            detection_rules = item.get("detection_rules", {})
            node_id = 0
            rule_keys = list(detection_rules.keys())

            impact = len(rule_keys) / 10.0  # normalize to [0,1]

            total_nodes = 0
            for key in rule_keys:
                keyword_list = detection_rules[key]
                type_code = node_type_map.get(key, 999)

                for kw in keyword_list:
                    G.add_node(node_id, keyword=kw, type=type_code, role=type_code)
                    if node_id > 0:
                        G.add_edge(node_id - 1, node_id, weight=0.6)
                    node_id += 1
                    total_nodes += 1

            likelihood = min(1.0, total_nodes / 20.0)

            severity = round(alpha * impact + (1 - alpha) * likelihood, 2)
            t_max = 5
            max_severity = max(max_severity, severity)

            patterns.append(
                {
                    "id": item["id"],
                    "name": item["name"],
                    "rule_keys": rule_keys,
                    "rule_dict": detection_rules,
                    "graph": G,
                    "severity": severity,
                    "t_max": t_max,
                }
            )

    return patterns, max_severity


def node_match(n1, n2):
    """节点匹配条件：类型和角色相同"""
    return n1["type"] == n2["type"]


def edge_match(e1, e2):
    """边匹配条件：权重满足阈值比例（θ_w=0.6）"""
    return e2["weight"] >= 0.6 * e1["weight"]


def calculate_ddrr(dialogue_graph, risk_patterns, max_severity=1.0, keyword_path=None):
    """计算DDRR指标"""
    numerator = 0.0
    denominator = 0.0
    keyword_data = json.load(open(keyword_path, "r", encoding="utf-8")) if keyword_path else {}
    matched_patterns = []

    for pattern in risk_patterns:
        Gm = pattern["graph"]
        t_max = pattern["t_max"]
        severity = pattern["severity"]
        rule_keys = pattern["rule_keys"]
        rule_dict = pattern["rule_dict"]
        print("\n正在匹配风险模式:", pattern["name"], " ", pattern["id"])

        # # 创建子图匹配器
        # gm = GraphMatcher(
        #     dialogue_graph, Gm, node_match=node_match, edge_match=edge_match
        # )

        # # 检查时间戳约束
        is_valid = False
        cnt = 0
        text_set = ""
        # for iso in gm.subgraph_isomorphisms_iter():
        #     timestamps = [dialogue_graph.nodes[n]["timestamp"] for n in iso.values()]
        #     if max(timestamps) <= t_max:
        #         is_valid = True
        #         break
        if isinstance(dialogue_graph, nx.Graph):
            for node, data in dialogue_graph.nodes(data=True):
                text_set += data["text"] + " "
        else:
            text_set = dialogue_graph
        for keyword_category in rule_keys:
            for keyword in rule_dict.get(keyword_category, []):
                is_matched = False
                for kw in keyword_data.get(keyword, []):
                    # 比较方法1：关键词直接匹配
                    # if kw in text_set:
                    #     is_matched = True
                    #     cnt += 1
                    #     break
                    # 比较方法2：WordNet同义词匹配
                    # if keyword_sentence_similarity_wordnet(kw, text_set) > 0.8:
                    #     is_matched = True
                    #     cnt += 1
                    #     break
                    # 比较方法3：向量相似度匹配
                    # if keyword_sentence_similarity_vector(kw, text_set, model_name=r"E:\\MAS\\model\\all-MiniLM-L6-v2") > 0.7:
                    #     is_matched = True
                    #     cnt += 1
                    #     break
                    # 比较方法4：使用快速相似度匹配器
                    # if quick_wordnet_match(kw, text_set) > 0.7:
                    #     is_matched = True
                    #     cnt += 1
                    #     break
                    # 比较方法5：使用句子相似度匹配
                    if quick_vector_match(kw, text_set) > 0.7:
                        is_matched = True
                        cnt += 1
                        break
                if is_matched:
                    break
        if cnt >= len(rule_keys):
            is_valid = True
            matched_patterns.append(pattern)
        # 更新分子分母
        weight = severity / max_severity
        denominator += weight
        if is_valid:
            numerator += weight

    print(f"匹配到 {len(matched_patterns)} 个风险模式")
    print(f"{' | '.join([p['name'] for p in matched_patterns])}")
    print(f"{' '.join([p['id'] for p in matched_patterns])}")
    ddrr_score = numerator / denominator if denominator != 0 else 0.0
    print(f"DDRR Score: {ddrr_score:.4f}\n")
    return ddrr_score, matched_patterns


def ddrr_validation(dialogue_graph, matched_patterns, risk_json_path, adversarial_samples_path=None, n_fragments=3, target_ddrr=0.8, max_iterations=100, risk_patterns=None, max_severity=1.0, keyword_path=None):
    text_set = ""
    matched_patterns_id = [x["id"] for x in matched_patterns]

    patterns_data = json.load(open(risk_json_path, "r", encoding="utf-8"))
    adversarial_samples = json.load(open(adversarial_samples_path, "r", encoding="utf-8"))
    id_list = []
    matched_list = [x["id"] for x in matched_patterns]
    for category, pattern_list in patterns_data.items():
        for j in range(len(pattern_list)):
            if pattern_list[j]["id"] not in matched_list:
                id_list.append(pattern_list[j]["id"])
    
    for node,data in dialogue_graph.nodes(data=True):
        text_set += data["text"] + " "
    choiced_patterns = list(combinations(id_list, n_fragments))
    adversarial_max_ddrr = 0.0
    for pattern in choiced_patterns:
        for id in pattern:
            sample = adversarial_samples.get(id, None)
            adversarial_text_set = text_set + sample[0]
        adversarial_ddrr, _ = calculate_ddrr(adversarial_text_set, risk_patterns=risk_patterns, max_severity=max_severity, keyword_path=keyword_path)
        adversarial_max_ddrr = max(adversarial_max_ddrr, adversarial_ddrr)
        if adversarial_max_ddrr > 0.8:
            return True, pattern
    print(f"对抗样本最大DDRR: {adversarial_max_ddrr:.4f}")
    return False, None

# ======================== RADI计算 ========================
# Complete implementation of calculate_radi and calculate_crra functions
def calculate_radi(risk_scores, timestamps, theta_r, conv_num):
    """
    Calculate Risk Assembly Delay Index (RADI)

    Formula: RADI = Σ[t_detect_k - t_start_k]+ / |R|
    where [x]+ = max(x, 0)

    Args:
        risk_scores: torch.Tensor - Node risk scores
        timestamps: np.array - Timestamps for each node
        theta_r: float - Adaptive threshold
        conv_num: int - Number of conversation rounds

    Returns:
        float: RADI score
    """
    if len(risk_scores) == 0:
        return 0.0

    # Convert risk_scores to numpy if it's a tensor
    if torch.is_tensor(risk_scores):
        risk_scores_np = risk_scores.detach().numpy()
    else:
        risk_scores_np = risk_scores

    # Find all risk events (nodes exceeding threshold)
    risk_indices = np.where(
        risk_scores_np >= theta_r.item() if torch.is_tensor(theta_r) else theta_r
    )[0]

    if len(risk_indices) == 0:
        return 0.0  # No risks detected

    # Group risk indices by conversation round (timestamp)
    risk_by_round = defaultdict(list)
    for idx in risk_indices:
        round_t = int(timestamps[idx])
        risk_by_round[round_t].append(idx)

    total_delay = 0.0
    detected_risks = 0

    # For each conversation round with risks
    for round_t, indices in risk_by_round.items():
        # t_start: earliest risk occurrence in this round
        t_start = round_t

        # t_detect: when the risk was first detected (threshold exceeded)
        # In this implementation, we assume detection happens in the same round
        # as the risk occurrence for simplicity
        t_detect = round_t

        # For more realistic simulation, we could add detection delay:
        # t_detect = round_t + detection_delay

        # Calculate delay for this risk group
        delay = max(0, t_detect - t_start)  # [x]+ = max(x, 0)
        total_delay += delay
        detected_risks += 1

    # Calculate average delay
    radi_score = total_delay / detected_risks if detected_risks > 0 else 0.0

    return radi_score


# Alternative implementation with sliding window for RADI (more realistic detection)
def calculate_radi_sliding_window(
    risk_scores, timestamps, theta_r, conv_num, window_size=3
):
    """
    Calculate RADI with sliding window detection simulation

    This version simulates more realistic risk detection by using a sliding window
    approach where risks might be detected with some delay.
    """
    if len(risk_scores) == 0:
        return 0.0

    # Convert to numpy
    if torch.is_tensor(risk_scores):
        risk_scores_np = risk_scores.detach().numpy()
    else:
        risk_scores_np = risk_scores

    if torch.is_tensor(theta_r):
        threshold = theta_r.item()
    else:
        threshold = theta_r

    # Simulate sliding window detection
    detected_delays = []

    for round_t in range(conv_num):
        # Find nodes in this round
        round_nodes = np.where(timestamps == round_t)[0]

        if len(round_nodes) == 0:
            continue

        # Check if any node in this round exceeds threshold
        round_risks = risk_scores_np[round_nodes]
        max_risk = np.max(round_risks)

        if max_risk >= threshold:
            # Risk started at round_t
            t_start = round_t

            # Simulate detection with sliding window
            t_detect = t_start
            for detection_round in range(t_start, min(t_start + window_size, conv_num)):
                # Check if risk is detected in this round
                detection_nodes = np.where(timestamps == detection_round)[0]
                if len(detection_nodes) > 0:
                    detection_risks = risk_scores_np[detection_nodes]
                    if np.max(detection_risks) >= threshold:
                        t_detect = detection_round
                        break

            # Calculate delay
            delay = max(0, t_detect - t_start)
            detected_delays.append(delay)

    # Calculate average delay
    if len(detected_delays) > 0:
        radi_score = np.mean(detected_delays)
    else:
        radi_score = 0.0

    return radi_score


def calculate_radi_with_detection_delay(risk_scores, timestamps, theta_r, conv_num, detection_delay=1):
    """
    计算带检测延迟的风险集成延迟指数(RADI)

    RADI公式: RADI = Σ[t_detect_k - t_start_k]+ / |R|
    其中 [x]+ = max(x, 0) 表示取正值

    Args:
        risk_scores: torch.Tensor 或 numpy.array - 每个节点的风险分数
        timestamps: numpy.array - 每个节点对应的时间戳(对话轮次)
        theta_r: float 或 torch.Tensor - 自适应风险阈值
        conv_num: int - 对话总轮次数
        detection_delay: int - 模拟的检测延迟轮次

    Returns:
        float: RADI分数，表示平均风险检测延迟
    """

    # ==================== 输入验证 ====================
    if len(risk_scores) == 0:
        # 如果没有风险分数数据，返回0（无风险）
        return 0.0

    # ==================== 数据类型转换 ====================
    # 将PyTorch张量转换为numpy数组，便于后续计算
    if torch.is_tensor(risk_scores):
        risk_scores_np = risk_scores.detach().numpy()  # detach()确保不影响梯度计算
    else:
        risk_scores_np = risk_scores

    # 提取阈值数值（处理PyTorch张量和普通数值两种情况）
    threshold = theta_r.item() if torch.is_tensor(theta_r) else theta_r

    # ==================== 按对话轮次分组节点 ====================
    # 使用defaultdict创建字典，key为对话轮次，value为该轮次的节点索引列表
    nodes_by_round = defaultdict(list)

    # 遍历所有时间戳，将节点按轮次分组
    for i, timestamp in enumerate(timestamps):
        round_t = int(timestamp)  # 确保时间戳为整数
        nodes_by_round[round_t].append(i)  # 将节点索引i添加到对应轮次round_t

    # ==================== 风险检测和延迟计算 ====================
    risk_delays = []  # 存储每个风险事件的检测延迟

    # 遍历所有对话轮次
    for round_t in range(conv_num):
        # 检查当前轮次是否有节点数据
        if round_t not in nodes_by_round:
            continue  # 如果当前轮次没有节点，跳过

        # 获取当前轮次的所有节点索引
        round_nodes = nodes_by_round[round_t]

        # 获取当前轮次所有节点的风险分数
        round_risks = risk_scores_np[round_nodes]

        # 检查当前轮次是否存在超过阈值的风险
        if np.max(round_risks) >= threshold:
            # ==================== 风险事件处理 ====================

            # t_start: 风险开始时间（风险首次出现的轮次）
            t_start = round_t

            # t_detect: 风险检测时间（考虑检测延迟）
            # 使用min()确保检测时间不超过对话总轮次
            t_detect = min(t_start + detection_delay, conv_num - 1)

            # 计算检测延迟：t_detect - t_start
            # 使用max(0, ...)确保延迟不为负数（符合RADI公式中的[x]+）
            delay = max(0, t_detect - t_start)

            # 将计算得到的延迟添加到延迟列表中
            risk_delays.append(delay)

    # ==================== 计算最终RADI分数 ====================
    #如果检测到风险事件，计算平均延迟；否则返回0
    if risk_delays:
        # RADI = 所有风险事件延迟的平均值
        return np.mean(risk_delays)
    else:
        # 如果没有检测到任何风险事件，返回0
        return 0.0
    # return np.mean(risk_delays)



def calculate_radi_with_sliding_window(risk_scores, timestamps, theta_r, conv_num,
                                       detection_delay=1, window_size=3, stride=1,
                                       aggregation='addition'):
    """
    使用固定大小滑动窗口计算带检测延迟的风险集成延迟指数(RADI)

    Args:
        risk_scores: torch.Tensor 或 numpy.array - 每个节点的风险分数
        timestamps: numpy.array - 每个节点对应的时间戳(对话轮次)
        theta_r: float 或 torch.Tensor - 自适应风险阈值
        conv_num: int - 对话总轮次数
        detection_delay: int - 模拟的检测延迟轮次
        window_size: int - 滑动窗口大小
        stride: int - 窗口滑动步长
        aggregation: str - 窗口内风险分数聚合方式 ('max', 'mean', 'median','addition')

    Returns:
        float: RADI分数，表示平均风险检测延迟
    """
    # ==================== 输入验证 ====================
    if len(risk_scores) == 0:
        return 0.0

    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size和stride必须大于0")

    # ================== 数据类型转换 ====================
    if torch.is_tensor(risk_scores):
        risk_scores_np = risk_scores.detach().numpy()
    else:
        risk_scores_np = risk_scores

    threshold = theta_r.item() if torch.is_tensor(theta_r) else theta_r

    # ==================== 按对话轮次分组和预处理 ====================
    # 创建每轮次的风险分数数组
    round_risks = np.zeros(conv_num)  # 初始化为0
    round_counts = np.zeros(conv_num)  # 记录每轮次的节点数量

    # 将节点风险分数按轮次聚合
    for i, timestamp in enumerate(timestamps):
        round_t = int(timestamp)
        if 0 <= round_t < conv_num:
            # 使用最大值聚合同一轮次的多个节点风险分数
            round_risks[round_t] = max(round_risks[round_t], risk_scores_np[i])
            round_counts[round_t] += 1

    # ==================== 滑动窗口风险检测 ====================
    risk_delays = []
    detected_rounds = set()  # 记录已检测的风险轮次，避免重复计算

    # 滑动窗口遍历
    for window_start in range(0, conv_num - window_size + 1, stride):
        window_end = window_start + window_size

        # 提取当前窗口的风险分数
        window_risks = round_risks[window_start:window_end]
        window_rounds = list(range(window_start, window_end))

        # 过滤掉没有数据的轮次
        valid_indices = [i for i, round_t in enumerate(window_rounds)
                         if round_counts[round_t] > 0]

        if not valid_indices:
            continue

        valid_risks = window_risks[valid_indices]
        valid_rounds = [window_rounds[i] for i in valid_indices]

        # ==================== 窗口内风险聚合 ====================
        if aggregation == 'max':
            window_risk_score = np.max(valid_risks)
        elif aggregation == 'mean':
            window_risk_score = np.mean(valid_risks)
        elif aggregation == 'median':
            window_risk_score = np.median(valid_risks)
        elif aggregation== 'addition':
            window_risk_score = np.sum(valid_risks)
        else:
            raise ValueError(f"不支持的聚合方式: {aggregation}")

        # ==================== 风险检测和延迟计算 ====================
        if window_risk_score >= threshold:
            # 找到窗口内第一个超过阈值的轮次作为风险开始时间
            risk_start_round = None
            for round_t in valid_rounds:
                if round_risks[round_t] >= threshold and round_t not in detected_rounds:
                    risk_start_round = round_t
                    break

            # 如果找到新的风险事件
            if risk_start_round is not None:
                # 标记为已检测
                detected_rounds.add(risk_start_round)

                # 计算检测时间（窗口结束时间 + 检测延迟）
                t_start = risk_start_round
                t_detect = min(window_end - 1 + detection_delay, conv_num - 1)

                # 计算延迟
                delay = max(0, t_detect - t_start)
                risk_delays.append(delay)

    # ==================== 处理可能遗漏的风险事件 ====================
    # 检查最后一个不完整窗口
    if conv_num % stride != 0:
        remaining_start = ((conv_num - window_size) // stride + 1) * stride
        if remaining_start < conv_num:
            for round_t in range(remaining_start, conv_num):
                if (round_counts[round_t] > 0 and
                        round_risks[round_t] >= threshold and
                        round_t not in detected_rounds):
                    t_start = round_t
                    t_detect = min(round_t + detection_delay, conv_num - 1)
                    delay = max(0, t_detect - t_start)
                    risk_delays.append(delay)

    # ==================== 计算最终RADI分数 ====================
    if risk_delays:
        return np.mean(risk_delays)
    else:
        return 0.0


# ======================== CRRA计算 ========================
def calculate_crra(V, edge_index, edge_attr, role_list, lambda_decay=0.2):
    """
    Calculate Cross-Role Risk Association (CRRA)

    Formula: CRRA = Σ_{(rp,rq)∈R²} φ(p,q) / (|R| choose 2)
    where φ(p,q) = Σ_{eij∈Epq} wij·γ(ti,tj) / √(Np·Nq)

    Args:
        V: torch.Tensor - Node features [timestamp, type, role, embedding...]
        edge_index: torch.Tensor - Edge connections [2, num_edges]
        edge_attr: torch.Tensor - Edge attributes [weight, delta_t, s_ij]
        role_list: list - List of unique roles
        lambda_decay: float - Decay coefficient (default ln(2)/3 ≈ 0.2)

    Returns:
        float: CRRA score
    """
    if len(role_list) <= 1:
        return 0.0  # Need at least 2 roles for cross-role analysis

    # Convert tensors to numpy for easier manipulation
    if torch.is_tensor(V):
        V_np = V.detach().numpy()
    else:
        V_np = V

    if torch.is_tensor(edge_index):
        edge_index_np = edge_index.detach().numpy()
    else:
        edge_index_np = edge_index

    if torch.is_tensor(edge_attr):
        edge_attr_np = edge_attr.detach().numpy()
    else:
        edge_attr_np = edge_attr

    # Count nodes per role
    role_counts = defaultdict(int)
    node_roles = {}  # node_idx -> role

    for i in range(V_np.shape[0]):
        role = int(V_np[i, 2])  # Role is at index 2
        role_counts[role] += 1
        node_roles[i] = role

    # Calculate φ(p,q) for each role pair
    phi_values = []
    role_pairs = []

    # Generate all unique role pairs
    unique_roles = list(set(role_list))
    for i, role_p in enumerate(unique_roles):
        for j, role_q in enumerate(unique_roles):
            if i <= j:  # Avoid duplicate pairs (p,q) and (q,p)
                role_pairs.append((role_p, role_q))

    for role_p, role_q in role_pairs:
        # Find edges between roles p and q (E_pq)
        cross_role_edges = []

        for edge_idx in range(edge_index_np.shape[1]):
            src_node = int(edge_index_np[0, edge_idx])
            dst_node = int(edge_index_np[1, edge_idx])

            src_role = node_roles.get(src_node, -1)
            dst_role = node_roles.get(dst_node, -1)

            # Check if this edge connects roles p and q
            if (src_role == role_p and dst_role == role_q) or (
                src_role == role_q and dst_role == role_p
            ):
                cross_role_edges.append(edge_idx)

        # Calculate φ(p,q)
        if len(cross_role_edges) == 0:
            phi_pq = 0.0
        else:
            weighted_sum = 0.0

            for edge_idx in cross_role_edges:
                # Get edge attributes
                w_ij = edge_attr_np[edge_idx, 0]  # Weight
                delta_t = edge_attr_np[edge_idx, 1]  # Time difference

                # Calculate time decay factor γ(ti, tj) = exp(-λ|ti - tj|)
                gamma = math.exp(-lambda_decay * abs(delta_t))

                # Add to weighted sum
                weighted_sum += w_ij * gamma

            # Normalize by sqrt(Np * Nq)
            N_p = role_counts[role_p]
            N_q = role_counts[role_q]

            if N_p > 0 and N_q > 0:
                phi_pq = weighted_sum / math.sqrt(N_p * N_q)
            else:
                phi_pq = 0.0

        phi_values.append(phi_pq)

    # Calculate CRRA
    if len(phi_values) == 0:
        return 0.0

    total_phi = sum(phi_values)
    num_role_pairs = len(role_pairs)

    # CRRA = Σφ(p,q) / (|R| choose 2)
    # Note: We use the actual number of role pairs instead of combinations
    # since we're considering all pairs including (p,p)
    crra_score = total_phi / num_role_pairs if num_role_pairs > 0 else 0.0

    return crra_score


def calculate_crra_ablated(V, edge_index, edge_attr, role_list, lambda_decay=0.2):
    """
    Calculate Cross-Role Risk Association (CRRA) in ablated mode
    
    In the ablated state, the role association analysis module is removed.
    This means:
    1. No role relationship extraction/construction
    2. No risk propagation calculation between roles
    3. Simplified risk assessment without cross-role interactions
    
    This function simulates the system without role association by:
    - Ignoring role information in edge calculations
    - Treating all nodes as having the same role
    - Removing cross-role weighted interactions

    Args:
        V: torch.Tensor - Node features [timestamp, type, role, embedding...]
        edge_index: torch.Tensor - Edge connections [2, num_edges]
        edge_attr: torch.Tensor - Edge attributes [weight, delta_t, s_ij]
        role_list: list - List of unique roles (ignored in ablated mode)
        lambda_decay: float - Decay coefficient (default ln(2)/3 ≈ 0.2)

    Returns:
        float: CRRA score in ablated mode (simplified calculation)
    """
    if len(role_list) <= 1:
        return 0.0  # Need at least 2 roles for cross-role analysis

    # Convert tensors to numpy for easier manipulation
    if torch.is_tensor(V):
        V_np = V.detach().numpy()
    else:
        V_np = V

    if torch.is_tensor(edge_index):
        edge_index_np = edge_index.detach().numpy()
    else:
        edge_index_np = edge_index

    if torch.is_tensor(edge_attr):
        edge_attr_np = edge_attr.detach().numpy()
    else:
        edge_attr_np = edge_attr

    # In ablated mode, treat all nodes as having a single role
    # This removes the role differentiation effect
    total_nodes = V_np.shape[0]
    
    # Calculate simplified φ without role-specific weighting
    # Sum all edge weights with time decay, but without role-based normalization
    total_weighted_sum = 0.0
    total_edges = edge_index_np.shape[1]
    
    if total_edges == 0:
        return 0.0
    
    for edge_idx in range(total_edges):
        # Get edge attributes
        w_ij = edge_attr_np[edge_idx, 0]  # Weight
        delta_t = edge_attr_np[edge_idx, 1]  # Time difference
        
        # Calculate time decay factor γ(ti, tj) = exp(-λ|ti - tj|)
        gamma = math.exp(-lambda_decay * abs(delta_t))
        
        # Add to weighted sum (without role-specific weighting)
        total_weighted_sum += w_ij * gamma
    
    # Simplified normalization: divide by total number of possible node pairs
    # This removes the role-specific √(Np·Nq) normalization
    max_possible_pairs = total_nodes * (total_nodes - 1) / 2
    
    if max_possible_pairs > 0:
        crra_ablated = total_weighted_sum / max_possible_pairs
    else:
        crra_ablated = 0.0
    
    return crra_ablated


def calculate_delta_crra(V, edge_index, edge_attr, role_list, lambda_decay=0.2):
    """
    Calculate Delta CRRA for ablation study
    
    Formula: Δ CRRA = (CRRA_full - CRRA_ablated) / CRRA_full
    
    This measures the importance of the role association analysis module
    by comparing the full system performance with the ablated version.
    
    Args:
        V: torch.Tensor - Node features [timestamp, type, role, embedding...]
        edge_index: torch.Tensor - Edge connections [2, num_edges]
        edge_attr: torch.Tensor - Edge attributes [weight, delta_t, s_ij]
        role_list: list - List of unique roles
        lambda_decay: float - Decay coefficient (default ln(2)/3 ≈ 0.2)
    
    Returns:
        dict: Contains CRRA_full, CRRA_ablated, and Delta_CRRA values
    """
    
    # Calculate CRRA in full mode (with role association analysis)
    crra_full = calculate_crra(V, edge_index, edge_attr, role_list, lambda_decay)
    
    # Calculate CRRA in ablated mode (without role association analysis)
    crra_ablated = calculate_crra_ablated(V, edge_index, edge_attr, role_list, lambda_decay)
    
    # Calculate Delta CRRA
    if crra_full == 0.0:
        # If CRRA_full is 0, Delta CRRA is undefined or 0
        delta_crra = 0.0
    else:
        delta_crra = (crra_full - crra_ablated) / crra_full
    
    return {
        "CRRA_full": crra_full,
        "CRRA_ablated": crra_ablated,
        "Delta_CRRA": delta_crra,
        "improvement_ratio": delta_crra * 100  # As percentage
    }


def analyze_role_association_importance(V, edge_index, edge_attr, role_list, lambda_decay=0.2):
    """
    Comprehensive analysis of role association module importance
    
    This function provides detailed insights into how the role association
    analysis contributes to the overall risk assessment system.
    
    Args:
        V: torch.Tensor - Node features [timestamp, type, role, embedding...]
        edge_index: torch.Tensor - Edge connections [2, num_edges]
        edge_attr: torch.Tensor - Edge attributes [weight, delta_t, s_ij]
        role_list: list - List of unique roles
        lambda_decay: float - Decay coefficient
    
    Returns:
        dict: Comprehensive analysis results
    """
    
    # Calculate delta CRRA
    delta_results = calculate_delta_crra(V, edge_index, edge_attr, role_list, lambda_decay)
    
    # Additional analysis
    unique_roles = list(set(role_list))
    num_roles = len(unique_roles)
    
    # Role distribution analysis
    if torch.is_tensor(V):
        V_np = V.detach().numpy()
    else:
        V_np = V
    
    role_distribution = {}
    for i in range(V_np.shape[0]):
        role = int(V_np[i, 2])
        role_distribution[role] = role_distribution.get(role, 0) + 1
    
    # Calculate role interaction density
    role_interaction_edges = 0
    same_role_edges = 0
    
    if torch.is_tensor(edge_index):
        edge_index_np = edge_index.detach().numpy()
    else:
        edge_index_np = edge_index
    
    for edge_idx in range(edge_index_np.shape[1]):
        src_node = int(edge_index_np[0, edge_idx])
        dst_node = int(edge_index_np[1, edge_idx])
        
        if src_node < len(role_list) and dst_node < len(role_list):
            src_role = int(V_np[src_node, 2]) if src_node < V_np.shape[0] else -1
            dst_role = int(V_np[dst_node, 2]) if dst_node < V_np.shape[0] else -1
            
            if src_role != -1 and dst_role != -1:
                if src_role != dst_role:
                    role_interaction_edges += 1
                else:
                    same_role_edges += 1
    
    total_edges = role_interaction_edges + same_role_edges
    cross_role_ratio = role_interaction_edges / total_edges if total_edges > 0 else 0.0
    
    # Determine the importance level
    delta_crra = delta_results["Delta_CRRA"]
    if delta_crra > 0.7:
        importance_level = "Critical"
    elif delta_crra > 0.4:
        importance_level = "High"
    elif delta_crra > 0.2:
        importance_level = "Moderate"
    elif delta_crra > 0.0:
        importance_level = "Low"
    else:
        importance_level = "None"
    
    return {
        **delta_results,
        "num_roles": num_roles,
        "role_distribution": role_distribution,
        "cross_role_edges": role_interaction_edges,
        "same_role_edges": same_role_edges,
        "cross_role_ratio": cross_role_ratio,
        "importance_level": importance_level,
        "analysis_summary": {
            "module_effectiveness": f"The role association analysis module contributes {delta_crra*100:.1f}% to the overall CRRA score",
            "cross_role_interactions": f"{cross_role_ratio*100:.1f}% of edges connect different roles",
            "recommendation": f"Role association analysis is {importance_level.lower()} importance for this conversation"
        }
    }

# ======================== GCRI计算 ========================
def calculate_trs(risk_scores, timestamps, lambda_decay=math.log(2) / 3.47, conv_num=None):
    """
    Calculate Temporal Risk Score (TRS)
    TRS = Σ(r_t * e^(-λ(T-t))) for t=1 to T
    """
    if isinstance(risk_scores, torch.Tensor):
        risk_scores = risk_scores.detach().numpy()
    if isinstance(timestamps, torch.Tensor):
        timestamps = timestamps.detach().numpy()
        
    # T = len(risk_scores)
    T = conv_num
    dialog_length = len(risk_scores) // T
    trs = 0.0

    for t in range(len(risk_scores)):
        decay_factor = math.exp(-lambda_decay * (T - (t // dialog_length) - 1)) # 需要修改
        trs += risk_scores[t] * decay_factor

    return trs


def build_risk_lexicon():
    """Build a weighted risk lexicon for CRD calculation"""
    risk_lexicon = {
        # Violence and harm - High Priority
        "kill": 1.0,
        "murder": 1.0,
        "death": 0.8,
        "violence": 0.9,
        "attack": 0.8,
        "weapon": 0.7,
        "gun": 0.8,
        "bomb": 1.0,
        "explosive": 0.9,
        "poison": 0.9,
        "harm": 0.7,
        "hurt": 0.6,
        "damage": 0.5,
        "destroy": 0.6,
        "threat": 0.8,
        "assassinate": 0.95,
        "massacre": 0.95,
        "genocide": 0.95,
        "torture": 0.9,
        "kidnap": 0.9,
        "hostage": 0.85,
        
        # Illegal activities - High Priority
        "drug": 1.0,
        "cocaine": 1.0,
        "heroin": 1.0,
        "fentanyl": 1.0,
        "methamphetamine": 0.95,
        "mdma": 0.85,
        "marijuana": 0.7,
        "steal": 0.8,
        "theft": 0.8,
        "fraud": 0.8,
        "scam": 0.8,
        "illegal": 0.7,
        "crime": 0.8,
        "criminal": 0.7,
        "hack": 0.8,
        "breach": 0.7,
        "piracy": 0.6,
        "trafficking": 0.9,
        "smuggling": 0.85,
        
        # Cybersecurity threats - High Priority
        "malware": 0.9,
        "virus": 0.8,
        "trojan": 0.9,
        "ransomware": 1.0,
        "keylogger": 0.8,
        "botnet": 0.9,
        "phishing": 0.8,
        "ddos": 0.7,
        "exploit": 0.8,
        "vulnerability": 0.7,
        "backdoor": 0.85,
        "rootkit": 0.85,
        
        # Explosives and weapons - Very High Priority
        "ied": 0.95,
        "improvised explosive device": 0.95,
        "c4": 0.9,
        "tnt": 0.9,
        "dynamite": 0.9,
        "grenade": 0.9,
        "ammonium nitrate": 0.9,
        "fertilizer bomb": 0.95,
        "pipe bomb": 0.9,
        "pressure cooker bomb": 0.95,
        "gunpowder": 0.85,
        "blasting cap": 0.95,
        "detonator": 0.95,
        "timer": 0.7,
        "remote control": 0.6,
        "trigger": 0.7,
        
        # Terrorism and extremism - Very High Priority
        "terrorism": 1.0,
        "terrorist": 1.0,
        "isis": 1.0,
        "al-qaeda": 1.0,
        "jihad": 0.9,
        "extremist": 0.8,
        "radical": 0.7,
        "recruitment": 0.8,
        "radicalization": 0.9,
        "martyrdom": 0.8,
        "propaganda": 0.7,
        
        # Suicide and self-harm - Very High Priority
        "suicide": 0.95,
        "self-harm": 0.9,
        "overdose": 0.9,
        "hanging": 0.95,
        "suffocation": 0.95,
        "cutting": 0.8,
        "pills": 0.6,
        "jump": 0.7,
        "bridge": 0.6,
        "despair": 0.7,
        "hopeless": 0.7,
        
        # Child exploitation - Very High Priority
        "child exploitation": 1.0,
        "child abuse": 0.95,
        "child pornography": 0.95,
        "grooming": 0.9,
        "predator": 0.85,
        "pedophile": 0.95,
        "minors": 0.7,
        "inappropriate": 0.6,
        "sexual abuse": 0.9,
        
        # Discrimination and hate - Medium-High Priority
        "hate": 0.8,
        "racist": 0.9,
        "discrimination": 0.8,
        "prejudice": 0.7,
        "stereotype": 0.6,
        "bias": 0.5,
        "sexist": 0.8,
        "homophobic": 0.9,
        "nazi": 0.9,
        "supremacist": 0.9,
        "fascist": 0.8,
        
        # Sexual content - Medium-High Priority
        "sexual": 0.7,
        "porn": 0.8,
        "pornography": 0.8,
        "explicit": 0.7,
        "nude": 0.7,
        "obscene": 0.8,
        "erotic": 0.6,
        "prostitution": 0.8,
        
        # Privacy violations - Medium Priority
        "private": 0.5,
        "personal": 0.4,
        "confidential": 0.6,
        "secret": 0.5,
        "password": 0.7,
        "account": 0.4,
        "identity": 0.5,
        "surveillance": 0.7,
        "tracking": 0.6,
        "monitoring": 0.6,
        
        # Financial crimes - High Priority
        "money laundering": 0.9,
        "ponzi": 0.9,
        "pyramid": 0.8,
        "embezzlement": 0.8,
        "bribery": 0.75,
        "corruption": 0.7,
        "tax evasion": 0.7,
        "shell company": 0.8,
        "offshore": 0.7,
        
        # General risk terms
        "dangerous": 0.7,
        "risky": 0.6,
        "unsafe": 0.6,
        "toxic": 0.8,
        "malicious": 0.8,
        "suspicious": 0.6,
        "warning": 0.5,
        "illegal": 0.7,
        "forbidden": 0.7,
        "prohibited": 0.7,
        "banned": 0.7,
        "restricted": 0.6,
    }
    return risk_lexicon


def calculate_crd_simple(conversation_texts, risk_lexicon):
    """Simple CRD calculation using word frequency"""
    combined_text = " ".join(conversation_texts).lower()
    words = combined_text.split()

    if not words:
        return 0.0

    word_counts = Counter(words)
    total_words = len(words)

    risk_score = 0.0
    for word, count in word_counts.items():
        if word in risk_lexicon:
            tf = count / total_words
            risk_weight = risk_lexicon[word]
            risk_score += tf * risk_weight

    return risk_score


def calculate_crd(conversation_texts, risk_lexicon=None):
    """Calculate Contextual Risk Density (CRD)"""
    if risk_lexicon is None:
        risk_lexicon = build_risk_lexicon()

    if TfidfVectorizer is None:
        return calculate_crd_simple(conversation_texts, risk_lexicon)

    combined_text = " ".join(conversation_texts)

    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform([combined_text])
        feature_names = vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
    except ValueError:
        return 0.0

    numerator = 0.0
    denominator_sum = 0.0

    for i, word in enumerate(feature_names):
        tfidf_score = tfidf_scores[i]
        denominator_sum += tfidf_score**2

        if word in risk_lexicon:
            risk_weight = risk_lexicon[word]
            numerator += tfidf_score * risk_weight

    if denominator_sum == 0:
        return 0.0

    crd = numerator / math.sqrt(denominator_sum)
    return crd


def calculate_gcri(
    risk_scores,
    timestamps,
    conversation_texts,
    alpha=0.5,
    beta=0.5,
    lambda_decay=math.log(2) / 3.47,
    conv_num=None,
    risk_lexicon_path=None
):
    """Calculate Global Conversation Risk Index (GCRI)"""
    # Ensure alpha + beta = 1
    total_weight = alpha + beta
    if abs(total_weight - 1.0) > 1e-6:
        alpha = alpha / total_weight
        beta = beta / total_weight

    # Calculate TRS
    trs = calculate_trs(risk_scores, timestamps, lambda_decay, conv_num)

    # Calculate CRD
    if os.path.exists(risk_lexicon_path):
        risk_lexicon = json.load(open(risk_lexicon_path, "r", encoding="utf-8"))
    else:
        risk_lexicon = None
    crd = calculate_crd(conversation_texts, risk_lexicon=risk_lexicon)

    # Calculate GCRI
    gcri = alpha * trs + beta * crd

    return {"GCRI": gcri, "TRS": trs, "CRD": crd, "alpha": alpha, "beta": beta}


def classify_risk_level(gcri_value):
    """Classify risk level based on GCRI value"""
    if gcri_value < 0.3:
        return {"level": "Safe", "action": "Allow", "color": "green"}
    elif gcri_value < 0.6:
        return {"level": "Low", "action": "Log", "color": "yellow"}
    elif gcri_value < 0.8:
        return {"level": "Medium", "action": "Review", "color": "orange"}
    else:
        return {"level": "High", "action": "Block", "color": "red"}
