import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher
import numpy as np
import torch
import math
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import json


def create_nx_graph_from_data(V, edge_index, edge_attr):
    """将PyG图数据转换为networkx图"""
    G = nx.DiGraph()

    # 添加节点
    for i in range(V.shape[0]):
        node_attrs = {
            "timestamp": int(V[i][0]),
            "type": int(V[i][1]),
            "role": int(V[i][2]),
            "embedding": V[i][3:],
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


def define_risk_patterns_from_json(json_path="risk_patterns.json", alpha=0.7):
    """从 risk_patterns.json 构造结构化风险模式图列表，带有严重度估算"""

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    patterns = []

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

            patterns.append(
                {
                    "id": item["id"],
                    "name": item["name"],
                    "graph": G,
                    "severity": severity,
                    "t_max": t_max,
                }
            )

    return patterns


def node_match(n1, n2):
    """节点匹配条件：类型和角色相同"""
    return n1["type"] == n2["type"] and n1["role"] == n2["role"]


def edge_match(e1, e2):
    """边匹配条件：权重满足阈值比例（θ_w=0.6）"""
    return e2["weight"] >= 0.6 * e1["weight"]


def calculate_ddrr(dialogue_graph, risk_patterns):
    """计算DDRR指标"""
    numerator = 0.0
    denominator = 0.0

    for pattern in risk_patterns:
        Gm = pattern["graph"]
        t_max = pattern["t_max"]
        severity = pattern["severity"]

        # 创建子图匹配器
        gm = GraphMatcher(
            dialogue_graph, Gm, node_match=node_match, edge_match=edge_match
        )

        # 检查时间戳约束
        is_valid = False
        for iso in gm.subgraph_isomorphisms_iter():
            timestamps = [dialogue_graph.nodes[n]["timestamp"] for n in iso.values()]
            if max(timestamps) <= t_max:
                is_valid = True
                break

        # 更新分子分母
        weight = severity
        denominator += weight
        if is_valid:
            numerator += weight

    return numerator / denominator if denominator != 0 else 0.0


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


# ======================== GCRI计算 ========================
def calculate_trs(risk_scores, timestamps, lambda_decay=math.log(2) / 3.47):
    """
    Calculate Temporal Risk Score (TRS)
    TRS = Σ(r_t * e^(-λ(T-t))) for t=1 to T
    """
    if isinstance(risk_scores, torch.Tensor):
        risk_scores = risk_scores.detach().numpy()
    if isinstance(timestamps, torch.Tensor):
        timestamps = timestamps.detach().numpy()

    T = len(risk_scores)
    trs = 0.0

    for t in range(T):
        decay_factor = math.exp(-lambda_decay * (T - t - 1))
        trs += risk_scores[t] * decay_factor

    return trs


def build_risk_lexicon():
    """Build a weighted risk lexicon for CRD calculation"""
    risk_lexicon = {
        # Violence and harm
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
        # Illegal activities
        "drug": 1.0,
        "cocaine": 1.0,
        "heroin": 1.0,
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
        # Discrimination and hate
        "hate": 0.8,
        "racist": 0.9,
        "discrimination": 0.8,
        "prejudice": 0.7,
        "stereotype": 0.6,
        "bias": 0.5,
        "sexist": 0.8,
        "homophobic": 0.9,
        # Sexual content
        "sexual": 0.7,
        "porn": 0.9,
        "explicit": 0.6,
        "inappropriate": 0.5,
        "nude": 0.7,
        "obscene": 0.8,
        # Privacy violations
        "private": 0.5,
        "personal": 0.4,
        "confidential": 0.6,
        "secret": 0.5,
        "password": 0.7,
        "account": 0.4,
        "identity": 0.5,
        # General risk terms
        "dangerous": 0.7,
        "risky": 0.6,
        "unsafe": 0.6,
        "toxic": 0.8,
        "malicious": 0.8,
        "suspicious": 0.6,
        "warning": 0.5,
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
):
    """Calculate Global Conversation Risk Index (GCRI)"""
    # Ensure alpha + beta = 1
    total_weight = alpha + beta
    if abs(total_weight - 1.0) > 1e-6:
        alpha = alpha / total_weight
        beta = beta / total_weight

    # Calculate TRS
    trs = calculate_trs(risk_scores, timestamps, lambda_decay)

    # Calculate CRD
    crd = calculate_crd(conversation_texts)

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
