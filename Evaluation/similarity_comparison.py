# !pip install -U sentence-transformers
import os
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import jieba

# --- 准备工作：下载必要的数据包 (首次使用时需要) ---
nltk.data.path.insert(0, r"E:\\MAS\\MultiAgentPolyDialogueAttack\\Evaluation\\nltk_data")  # 确保 NLTK 数据包下载到当前目录
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')

def get_wordnet_pos(word):
    """将 NLTK 的词性标注映射到 WordNet 可接受的格式"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def keyword_sentence_similarity_wordnet(keyword, sentence):
    """
    使用 NLTK 和 WordNet 计算关键词与句子的相似度 - 优化版本。
    支持单个单词和词组作为关键词。
    返回 0 到 1 之间的分数。
    """
    # 1. 初始化和预处理
    stop_words = set(stopwords.words('english'))
    sentence_tokens = word_tokenize(sentence.lower())
    filtered_sentence_tokens = [w for w in sentence_tokens if not w in stop_words and w.isalpha()]
    
    # 2. 处理关键词（可能是单词或词组）
    keyword_tokens = word_tokenize(keyword.lower())
    keyword_tokens = [w for w in keyword_tokens if not w in stop_words and w.isalpha()]
    
    if not keyword_tokens or not filtered_sentence_tokens:
        return 0.0
    
    # 3. 精确匹配检查
    exact_match_score = check_exact_matches(keyword_tokens, filtered_sentence_tokens)
    if exact_match_score > 0:
        return min(1.0, exact_match_score + 0.1)  # 给精确匹配加分
    
    # 4. 如果关键词是词组，使用多种策略
    if len(keyword_tokens) > 1:
        return calculate_phrase_similarity(keyword_tokens, filtered_sentence_tokens)
    else:
        # 5. 单个关键词的处理（增强版）
        return calculate_single_word_similarity(keyword_tokens[0], filtered_sentence_tokens)

def check_exact_matches(keyword_tokens, sentence_tokens):
    """
    检查精确匹配和部分匹配
    """
    keyword_set = set(keyword_tokens)
    sentence_set = set(sentence_tokens)
    
    # 完全匹配
    exact_matches = keyword_set.intersection(sentence_set)
    if exact_matches:
        match_ratio = len(exact_matches) / len(keyword_set)
        return match_ratio
    
    return 0.0

def calculate_phrase_similarity(keyword_tokens, sentence_tokens):
    """
    计算词组与句子的相似度 - 优化版本
    """
    max_similarity = 0.0
    
    # 策略1: 滑动窗口完整匹配
    phrase_similarity = calculate_sliding_window_similarity(keyword_tokens, sentence_tokens)
    max_similarity = max(max_similarity, phrase_similarity)
    
    # 策略2: 词组中每个词的最佳匹配平均值
    individual_similarities = []
    for keyword_word in keyword_tokens:
        word_max_sim = 0.0
        for sentence_word in sentence_tokens:
            word_sim = calculate_word_similarity(keyword_word, sentence_word)
            word_max_sim = max(word_max_sim, word_sim)
        individual_similarities.append(word_max_sim)
    
    if individual_similarities:
        avg_similarity = sum(individual_similarities) / len(individual_similarities)
        max_similarity = max(max_similarity, avg_similarity)
    
    # 策略3: 词组中关键词的加权匹配
    weighted_similarity = calculate_weighted_phrase_similarity(keyword_tokens, sentence_tokens)
    max_similarity = max(max_similarity, weighted_similarity)
    
    # 策略4: 语义相关性检查
    semantic_similarity = calculate_semantic_phrase_similarity(keyword_tokens, sentence_tokens)
    max_similarity = max(max_similarity, semantic_similarity)
    
    return max_similarity

def calculate_sliding_window_similarity(keyword_tokens, sentence_tokens):
    """
    滑动窗口匹配 - 优化版本
    """
    if len(keyword_tokens) > len(sentence_tokens):
        return 0.0
    
    max_window_similarity = 0.0
    
    # 完整词组匹配检查
    sentence_text = ' '.join(sentence_tokens)
    keyword_phrase = ' '.join(keyword_tokens)
    if keyword_phrase in sentence_text:
        return 1.0
    
    # 滑动窗口匹配
    for i in range(len(sentence_tokens) - len(keyword_tokens) + 1):
        window = sentence_tokens[i:i + len(keyword_tokens)]
        window_similarity = 0.0
        
        # 计算窗口内每个词与关键词对应词的相似度
        for j, keyword_word in enumerate(keyword_tokens):
            if j < len(window):
                word_similarity = calculate_word_similarity(keyword_word, window[j])
                window_similarity += word_similarity
        
        # 取平均相似度，但给高相似度更多权重
        if len(keyword_tokens) > 0:
            avg_sim = window_similarity / len(keyword_tokens)
            # 如果平均相似度很高，给额外奖励
            if avg_sim > 0.7:
                avg_sim = min(1.0, avg_sim * 1.2)
            max_window_similarity = max(max_window_similarity, avg_sim)
    
    return max_window_similarity

def calculate_weighted_phrase_similarity(keyword_tokens, sentence_tokens):
    """
    加权词组相似度计算
    """
    # 给不同位置的词不同权重（通常第一个和最后一个词更重要）
    weights = []
    n = len(keyword_tokens)
    
    if n == 1:
        weights = [1.0]
    elif n == 2:
        weights = [0.6, 0.4]  # 第一个词权重更高
    else:
        # 首末词权重高，中间词权重低
        weights = [0.4] + [0.2] * (n-2) + [0.4]
        # 归一化权重
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
    
    weighted_similarity = 0.0
    for i, keyword_word in enumerate(keyword_tokens):
        word_max_sim = 0.0
        for sentence_word in sentence_tokens:
            word_sim = calculate_word_similarity(keyword_word, sentence_word)
            word_max_sim = max(word_max_sim, word_sim)
        
        weighted_similarity += word_max_sim * weights[i]
    
    return weighted_similarity

def calculate_semantic_phrase_similarity(keyword_tokens, sentence_tokens):
    """
    语义级别的词组相似度
    """
    max_semantic_sim = 0.0
    
    # 检查是否有语义相关的词汇群
    for keyword_word in keyword_tokens:
        semantic_group_sim = 0.0
        related_count = 0
        
        for sentence_word in sentence_tokens:
            word_sim = calculate_word_similarity(keyword_word, sentence_word)
            if word_sim > 0.3:  # 有一定相关性的词
                semantic_group_sim += word_sim
                related_count += 1
        
        if related_count > 0:
            avg_semantic_sim = semantic_group_sim / related_count
            # 如果有多个相关词，给额外奖励
            if related_count > 1:
                avg_semantic_sim = min(1.0, avg_semantic_sim * (1 + related_count * 0.1))
            max_semantic_sim = max(max_semantic_sim, avg_semantic_sim)
    
    return max_semantic_sim

def calculate_single_word_similarity(keyword_word, sentence_tokens):
    """
    单个关键词与句子的相似度计算 - 优化版本
    """
    similarities = []
    
    # 与每个句子词汇计算相似度
    for token in sentence_tokens:
        similarity = calculate_word_similarity(keyword_word, token)
        similarities.append(similarity)
    
    if not similarities:
        return 0.0
    
    # 使用多种策略组合
    max_sim = max(similarities)
    avg_sim = sum(similarities) / len(similarities)
    
    # 计算高相似度词汇的比例
    high_sim_count = sum(1 for sim in similarities if sim > 0.5)
    high_sim_ratio = high_sim_count / len(similarities)
    
    # 综合考虑最高相似度、平均相似度和高相似度比例
    final_similarity = max_sim * 0.7 + avg_sim * 0.2 + high_sim_ratio * 0.1
    
    return min(1.0, final_similarity)

def calculate_word_similarity(word1, word2):
    """
    计算两个单词之间的相似度 - 优化版本
    使用多种相似度计算方法以提高匹配率
    """
    if word1 == word2:
        return 1.0
    
    # 字符串相似度检查（处理拼写变化、复数等）
    string_sim = calculate_string_similarity(word1, word2)
    if string_sim > 0.8:  # 高字符串相似度直接返回
        return string_sim
    
    # 获取两个词的 synset
    word1_synsets = wordnet.synsets(word1)  # 不限制词性，获取所有可能的synset
    word2_synsets = wordnet.synsets(word2)
    
    if not word1_synsets or not word2_synsets:
        return string_sim  # 如果找不到synset，返回字符串相似度
    
    max_similarity = 0.0
    
    # 使用多种相似度计算方法
    for synset1 in word1_synsets:
        for synset2 in word2_synsets:
            similarities = []
            
            # 1. 路径相似度
            path_sim = synset1.path_similarity(synset2)
            if path_sim:
                similarities.append(path_sim)
            
            # 2. Wu-Palmer相似度（基于深度）
            wup_sim = synset1.wup_similarity(synset2)
            if wup_sim:
                similarities.append(wup_sim)
            
            # 3. Lin相似度（需要信息内容）
            try:
                lin_sim = synset1.lin_similarity(synset2, wordnet.ic.brown_ic)
                if lin_sim:
                    similarities.append(lin_sim)
            except:
                pass
            
            # 4. Jiang-Conrath距离转换为相似度
            try:
                jcn_dist = synset1.jcn_similarity(synset2, wordnet.ic.brown_ic)
                if jcn_dist:
                    similarities.append(jcn_dist)
            except:
                pass
            
            # 5. Resnik相似度
            try:
                res_sim = synset1.res_similarity(synset2, wordnet.ic.brown_ic)
                if res_sim:
                    # Resnik返回信息内容，需要归一化
                    normalized_res = min(res_sim / 10.0, 1.0)
                    similarities.append(normalized_res)
            except:
                pass
            
            # 取最大相似度
            if similarities:
                current_max = max(similarities)
                max_similarity = max(max_similarity, current_max)
    
    # 同义词和反义词检查
    synonym_sim = check_synonyms_antonyms(word1, word2)
    max_similarity = max(max_similarity, synonym_sim)
    
    # 词根检查
    stem_sim = check_word_stems(word1, word2)
    max_similarity = max(max_similarity, stem_sim)
    
    # 结合字符串相似度
    final_similarity = max(max_similarity, string_sim)
    
    return final_similarity

def calculate_string_similarity(word1, word2):
    """
    计算字符串相似度（编辑距离）
    """
    if word1 == word2:
        return 1.0
    
    # 计算编辑距离
    def edit_distance(s1, s2):
        if len(s1) < len(s2):
            return edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    max_len = max(len(word1), len(word2))
    if max_len == 0:
        return 1.0
    
    distance = edit_distance(word1.lower(), word2.lower())
    similarity = 1.0 - (distance / max_len)
    
    return max(0.0, similarity)

def check_synonyms_antonyms(word1, word2):
    """
    检查同义词和反义词关系
    """
    max_sim = 0.0
    
    for synset1 in wordnet.synsets(word1):
        for synset2 in wordnet.synsets(word2):
            # 检查是否为同义词（同一个synset）
            if synset1 == synset2:
                return 1.0
            
            # 检查上位词/下位词关系
            if synset2 in synset1.hyponyms() or synset1 in synset2.hyponyms():
                max_sim = max(max_sim, 0.8)
            
            if synset2 in synset1.hypernyms() or synset1 in synset2.hypernyms():
                max_sim = max(max_sim, 0.7)
            
            # 检查部分-整体关系
            if synset2 in synset1.part_holonyms() or synset1 in synset2.part_holonyms():
                max_sim = max(max_sim, 0.6)
            
            if synset2 in synset1.part_meronyms() or synset1 in synset2.part_meronyms():
                max_sim = max(max_sim, 0.6)
            
            # 检查相似关系
            if synset2 in synset1.similar_tos() or synset1 in synset2.similar_tos():
                max_sim = max(max_sim, 0.8)
    
    return max_sim

def check_word_stems(word1, word2):
    """
    检查词根相似度
    """
    try:
        from nltk.stem import PorterStemmer, WordNetLemmatizer
        
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        
        # 词干提取
        stem1 = stemmer.stem(word1)
        stem2 = stemmer.stem(word2)
        
        if stem1 == stem2:
            return 0.9
        
        # 词形还原
        lemma1 = lemmatizer.lemmatize(word1)
        lemma2 = lemmatizer.lemmatize(word2)
        
        if lemma1 == lemma2:
            return 0.85
        
        # 检查词根的字符串相似度
        stem_sim = calculate_string_similarity(stem1, stem2)
        lemma_sim = calculate_string_similarity(lemma1, lemma2)
        
        return max(stem_sim * 0.8, lemma_sim * 0.8)
        
    except ImportError:
        # 如果没有相关库，返回0
        return 0.0

# 添加中文分词支持
def is_chinese(text):
    """检查文本是否包含中文字符"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def tokenize_text(text):
    """
    根据语言自动选择分词方式
    """
    if is_chinese(text):
        # 中文分词
        return list(jieba.cut(text.strip()))
    else:
        # 英文分词
        return word_tokenize(text.lower())

def keyword_sentence_similarity_universal(keyword, sentence, method='vector', model_name=r'E:\\MAS\\model\\all-MiniLM-L6-v2'):
    """
    通用的相似度计算函数，支持中英文
    
    Args:
        keyword: 关键词（单词或词组，中英文均可）
        sentence: 目标句子（中英文均可）
        method: 计算方法，'vector' 或 'wordnet'
        model_name: 向量模型路径
    
    Returns:
        相似度分数 (0-1)
    """
    # 检查是否包含中文
    if is_chinese(keyword) or is_chinese(sentence):
        # 中文文本主要使用向量方法
        if method == 'wordnet':
            print("Warning: WordNet method is not suitable for Chinese text, switching to vector method.")
            method = 'vector'
    
    if method == 'vector':
        return keyword_sentence_similarity_vector(keyword, sentence, model_name)
    elif method == 'wordnet':
        if is_chinese(keyword) or is_chinese(sentence):
            # 对于中文，使用简单的字符串匹配作为fallback
            return simple_chinese_similarity(keyword, sentence)
        else:
            return keyword_sentence_similarity_wordnet(keyword, sentence)
    else:
        raise ValueError("method must be 'vector' or 'wordnet'")

def simple_chinese_similarity(keyword, sentence):
    """
    简单的中文相似度计算（基于词汇重叠）
    """
    keyword_tokens = set(tokenize_text(keyword))
    sentence_tokens = set(tokenize_text(sentence))
    
    # 计算关键词在句子中的覆盖率
    if not keyword_tokens:
        return 0.0
    
    # 完全匹配
    if keyword.strip() in sentence:
        return 1.0
    
    # 词汇重叠相似度
    intersection = keyword_tokens.intersection(sentence_tokens)
    if len(intersection) == 0:
        return 0.0
    
    # Jaccard相似度
    union = keyword_tokens.union(sentence_tokens)
    jaccard_sim = len(intersection) / len(union)
    
    # 关键词覆盖率
    keyword_coverage = len(intersection) / len(keyword_tokens)
    
    # 返回两者的最大值
    return max(jaccard_sim, keyword_coverage * 0.8)



def keyword_sentence_similarity_vector(keyword, sentence, model_name='all-MiniLM-L6-v2'):
    """
    使用 Sentence-Transformers 模型计算关键词与句子的向量相似度。
    支持单个单词和词组作为关键词。
    """
    model = SentenceTransformer(model_name)

    # 1. 将关键词和句子编码成向量
    keyword_embedding = model.encode(keyword, convert_to_tensor=True)
    sentence_embedding = model.encode(sentence, convert_to_tensor=True)

    # 2. 计算余弦相似度
    cosine_similarity = util.cos_sim(keyword_embedding, sentence_embedding)
    
    # 3. 如果关键词是词组，还可以计算词组中各个词与句子的相似度
    keyword_tokens = word_tokenize(keyword.lower())
    if len(keyword_tokens) > 1:
        # 计算关键词中每个词与句子的相似度，取最大值
        max_token_similarity = 0.0
        for token in keyword_tokens:
            if token.isalpha():  # 只处理字母词
                token_embedding = model.encode(token, convert_to_tensor=True)
                token_similarity = util.cos_sim(token_embedding, sentence_embedding)
                max_token_similarity = max(max_token_similarity, token_similarity.item())
        
        # 返回整个词组相似度和单词最大相似度的较高值
        return max(cosine_similarity.item(), max_token_similarity)
    
    return cosine_similarity.item()


# 添加一个通用的相似度计算函数
def calculate_similarity(keyword, sentence, method='vector', model_name=r'E:\\MAS\\model\\all-MiniLM-L6-v2'):
    """
    通用的相似度计算函数
    
    Args:
        keyword: 关键词（单词或词组）
        sentence: 目标句子
        method: 计算方法，'vector' 或 'wordnet'
        model_name: 向量模型路径（仅在method='vector'时使用）
    
    Returns:
        相似度分数 (0-1)
    """
    if method == 'vector':
        return keyword_sentence_similarity_vector(keyword, sentence, model_name)
    elif method == 'wordnet':
        return keyword_sentence_similarity_wordnet(keyword, sentence)
    else:
        raise ValueError("method must be 'vector' or 'wordnet'")

def test():
    # --- 优化后的WordNet测试 ---
    print("=== 优化后的WordNet相似度测试 ===")
    
    # 测试1: 同义词测试
    print("\n--- 同义词测试 ---")
    synonym_tests = [
        ("car", "I saw a green automobile on the street."),
        ("big", "The huge elephant walked slowly."),
        ("happy", "She felt joyful after the good news."),
        ("smart", "He is very intelligent and clever."),
        ("weapon", "The soldier carried his gun carefully."),
    ]
    
    for keyword, sentence in synonym_tests:
        similarity = keyword_sentence_similarity_wordnet(keyword, sentence)
        print(f"关键词 '{keyword}' 与句子 '{sentence}' 的相似度: {similarity:.4f}")
    
    # 测试2: 词组测试
    print("\n--- 词组测试 ---")
    phrase_tests = [
        ("sports car", "I bought a new racing vehicle yesterday."),
        ("sports car", "The red sports automobile is very fast."),
        ("explosive device", "The bomb squad found a dangerous package."),
        ("machine learning", "AI algorithms use neural networks for training."),
        ("network security", "The firewall protects against cyber attacks."),
        ("illegal drugs", "Police found narcotics in the building."),
    ]
    
    for keyword_phrase, sentence in phrase_tests:
        similarity = keyword_sentence_similarity_wordnet(keyword_phrase, sentence)
        print(f"关键词词组 '{keyword_phrase}' 与句子 '{sentence}' 的相似度: {similarity:.4f}")
    
    # 测试3: 复杂语义关系测试
    print("\n--- 复杂语义关系测试 ---")
    complex_tests = [
        ("violence", "The aggressive behavior led to physical confrontation."),
        ("hacking", "Cybercriminals broke into the computer system."),
        ("privacy", "Personal information was leaked to unauthorized parties."),
        ("terrorism", "The extremist group planned violent attacks."),
        ("fraud", "The company was involved in financial deception."),
        ("assassination", "The political leader was killed by a sniper."),
    ]
    
    for keyword, sentence in complex_tests:
        similarity = keyword_sentence_similarity_wordnet(keyword, sentence)
        print(f"关键词 '{keyword}' 与句子 '{sentence}' 的相似度: {similarity:.4f}")
    
    # 测试4: 不相关内容测试（应该得到低分）
    print("\n--- 不相关内容测试 ---")
    irrelevant_tests = [
        ("weapon", "I like eating ice cream on sunny days."),
        ("explosive device", "The beautiful flowers bloom in spring."),
        ("hacking", "She enjoys reading romantic novels."),
        ("violence", "The peaceful sunset was very calming."),
    ]
    
    for keyword, sentence in irrelevant_tests:
        similarity = keyword_sentence_similarity_wordnet(keyword, sentence)
        print(f"关键词 '{keyword}' 与句子 '{sentence}' 的相似度: {similarity:.4f}")
    
    # 测试5: 词形变化测试
    print("\n--- 词形变化测试 ---")
    morphology_tests = [
        ("kill", "The murderer killed three people."),
        ("kill", "This software can eliminate the virus."),
        ("explosive", "The bomb exploded in the building."),
        ("attack", "The hacker is attacking our servers."),
        ("steal", "The thief stole valuable items."),
        ("dangerous", "This chemical poses serious dangers."),
    ]
    
    for keyword, sentence in morphology_tests:
        similarity = keyword_sentence_similarity_wordnet(keyword, sentence)
        print(f"关键词 '{keyword}' 与句子 '{sentence}' 的相似度: {similarity:.4f}")
    
    # 比较优化前后的效果
    print("\n" + "="*60)
    print("=== 方法对比测试 ===")
    
    comparison_tests = [
        ("weapon", "The soldier carried his rifle."),
        ("explosive device", "The bomb disposal team worked carefully."),
        ("cyber attack", "Hackers infiltrated the network systems."),
        ("illegal substances", "Police discovered drugs in the warehouse."),
        ("violent behavior", "The aggressive actions caused injuries."),
    ]
    
    for keyword, sentence in comparison_tests:
        wordnet_sim = keyword_sentence_similarity_wordnet(keyword, sentence)
        vector_sim = keyword_sentence_similarity_vector(keyword, sentence, r'E:\\MAS\\model\\all-MiniLM-L6-v2')
        
        print(f"\n测试: '{keyword}' vs '{sentence}'")
        print(f"  优化WordNet方法: {wordnet_sim:.4f}")
        print(f"  向量方法: {vector_sim:.4f}")
        print(f"  差异: {abs(wordnet_sim - vector_sim):.4f}")
    
    print("\n" + "="*60)
    print("测试完成 - WordNet方法已优化")
    print("="*60)