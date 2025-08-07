# Fast Similarity Matcher - 简化版本，优化运行速度
import os
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import jieba
from functools import lru_cache

# NLTK数据路径配置
nltk.data.path.insert(0, r"E:\\MAS\\MultiAgentPolyDialogueAttack\\Evaluation\\nltk_data")

class FastSimilarityMatcher:
    """
    快速相似度匹配器 - 简化版本
    专注于速度优化，使用缓存和简化的匹配规则
    """
    
    def __init__(self, model_path=r'E:\\MAS\\model\\all-MiniLM-L6-v2'):
        """初始化匹配器"""
        self.model = None
        self.model_path = model_path
        self.stop_words = set(stopwords.words('english'))
        self._embedding_cache = {}  # 向量缓存
        self._wordnet_cache = {}    # WordNet缓存
        
    def _lazy_load_model(self):
        """延迟加载向量模型"""
        if self.model is None:
            self.model = SentenceTransformer(self.model_path)
    
    @lru_cache(maxsize=1000)
    def _preprocess_text(self, text):
        """预处理文本，带缓存"""
        tokens = word_tokenize(text.lower())
        return [w for w in tokens if not w in self.stop_words and w.isalpha()]
    
    def fast_wordnet_similarity(self, keyword, sentence, threshold=0.3):
        """
        快速WordNet相似度计算 - 简化版本
        使用缓存和早期退出策略
        """
        # 缓存键
        cache_key = f"{keyword}||{sentence}"
        if cache_key in self._wordnet_cache:
            return self._wordnet_cache[cache_key]
        
        # 预处理
        keyword_tokens = self._preprocess_text(keyword)
        sentence_tokens = self._preprocess_text(sentence)
        
        if not keyword_tokens or not sentence_tokens:
            self._wordnet_cache[cache_key] = 0.0
            return 0.0
        
        # 1. 快速精确匹配检查
        exact_score = self._quick_exact_match(keyword_tokens, sentence_tokens)
        if exact_score > 0.8:  # 早期退出
            self._wordnet_cache[cache_key] = exact_score
            return exact_score
        
        # 2. 简化的语义匹配
        if len(keyword_tokens) == 1:
            # 单词匹配
            semantic_score = self._fast_single_word_match(keyword_tokens[0], sentence_tokens)
        else:
            # 词组匹配
            semantic_score = self._fast_phrase_match(keyword_tokens, sentence_tokens)
        
        final_score = max(exact_score, semantic_score)
        self._wordnet_cache[cache_key] = final_score
        return final_score
    
    def _quick_exact_match(self, keyword_tokens, sentence_tokens):
        """快速精确匹配"""
        keyword_set = set(keyword_tokens)
        sentence_set = set(sentence_tokens)
        
        matches = keyword_set.intersection(sentence_set)
        if matches:
            return len(matches) / len(keyword_set)
        return 0.0
    
    @lru_cache(maxsize=500)
    def _get_word_synsets(self, word):
        """获取词的同义词集，带缓存"""
        return wordnet.synsets(word)
    
    def _fast_single_word_match(self, keyword, sentence_tokens):
        """快速单词匹配 - 只使用最常用的相似度方法"""
        max_sim = 0.0
        
        # 获取关键词的同义词集
        keyword_synsets = self._get_word_synsets(keyword)
        if not keyword_synsets:
            return 0.0
        
        # 只取前2个最相关的synset以提高速度
        keyword_synsets = keyword_synsets[:2]
        
        for token in sentence_tokens:
            if max_sim > 0.9:  # 早期退出
                break
                
            token_synsets = self._get_word_synsets(token)
            if not token_synsets:
                continue
            
            # 只取前2个synset
            token_synsets = token_synsets[:2]
            
            # 使用最快的相似度方法
            for k_syn in keyword_synsets:
                for t_syn in token_synsets:
                    # 只使用路径相似度（最快）
                    sim = k_syn.path_similarity(t_syn)
                    if sim:
                        max_sim = max(max_sim, sim)
                        if max_sim > 0.9:  # 早期退出
                            return max_sim
        
        return max_sim
    
    def _fast_phrase_match(self, keyword_tokens, sentence_tokens):
        """快速词组匹配 - 简化策略"""
        # 策略1: 检查是否词组完整出现
        sentence_text = ' '.join(sentence_tokens)
        keyword_phrase = ' '.join(keyword_tokens)
        if keyword_phrase in sentence_text:
            return 1.0
        
        # 策略2: 计算词组中每个词的最佳匹配（简化版）
        total_sim = 0.0
        for keyword_word in keyword_tokens:
            word_max_sim = 0.0
            for sentence_word in sentence_tokens:
                # 使用简化的词相似度计算
                sim = self._simple_word_similarity(keyword_word, sentence_word)
                word_max_sim = max(word_max_sim, sim)
                if word_max_sim > 0.9:  # 早期退出
                    break
            total_sim += word_max_sim
        
        return total_sim / len(keyword_tokens) if keyword_tokens else 0.0
    
    @lru_cache(maxsize=1000)
    def _simple_word_similarity(self, word1, word2):
        """简化的词相似度计算"""
        if word1 == word2:
            return 1.0
        
        # 简单的字符串相似度检查
        if self._quick_string_sim(word1, word2) > 0.8:
            return 0.9
        
        # 只使用最基本的WordNet匹配
        synsets1 = self._get_word_synsets(word1)[:1]  # 只取第一个
        synsets2 = self._get_word_synsets(word2)[:1]  # 只取第一个
        
        if synsets1 and synsets2:
            sim = synsets1[0].path_similarity(synsets2[0])
            return sim if sim else 0.0
        
        return 0.0
    
    def _quick_string_sim(self, word1, word2):
        """快速字符串相似度计算"""
        if len(word1) == 0 or len(word2) == 0:
            return 0.0
        
        # 简单的包含关系检查
        shorter, longer = (word1, word2) if len(word1) < len(word2) else (word2, word1)
        if shorter in longer:
            return len(shorter) / len(longer)
        
        # 简化的编辑距离计算
        if abs(len(word1) - len(word2)) > 3:  # 长度差异太大直接返回0
            return 0.0
        
        # 计算公共字符比例
        common = set(word1) & set(word2)
        total = set(word1) | set(word2)
        return len(common) / len(total) if total else 0.0
    
    def fast_vector_similarity(self, keyword, sentence):
        """
        快速向量相似度计算 - 使用缓存优化
        """
        # 检查缓存
        cache_key = f"vec_{keyword}||{sentence}"
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        # 延迟加载模型
        self._lazy_load_model()
        
        # 计算向量相似度
        keyword_embedding = self.model.encode(keyword, convert_to_tensor=True)
        sentence_embedding = self.model.encode(sentence, convert_to_tensor=True)
        
        similarity = util.cos_sim(keyword_embedding, sentence_embedding).item()
        
        # 缓存结果
        self._embedding_cache[cache_key] = similarity
        return similarity
    
    def batch_vector_similarity(self, keywords, sentences):
        """
        批量向量相似度计算 - 提高效率
        """
        self._lazy_load_model()
        
        # 批量编码
        keyword_embeddings = self.model.encode(keywords, convert_to_tensor=True)
        sentence_embeddings = self.model.encode(sentences, convert_to_tensor=True)
        
        # 计算相似度矩阵
        similarities = util.cos_sim(keyword_embeddings, sentence_embeddings)
        
        return similarities.cpu().numpy()
    
    def clear_cache(self):
        """清空缓存"""
        self._embedding_cache.clear()
        self._wordnet_cache.clear()
        self._preprocess_text.cache_clear()
        self._get_word_synsets.cache_clear()
        self._simple_word_similarity.cache_clear()


# 便捷函数
def quick_wordnet_match(keyword, sentence, matcher=None):
    """快速WordNet匹配的便捷函数"""
    if matcher is None:
        matcher = FastSimilarityMatcher()
    return matcher.fast_wordnet_similarity(keyword, sentence)

def quick_vector_match(keyword, sentence, matcher=None):
    """快速向量匹配的便捷函数"""
    if matcher is None:
        matcher = FastSimilarityMatcher()
    return matcher.fast_vector_similarity(keyword, sentence)


def benchmark_test():
    """性能基准测试"""
    import time
    
    print("=== 快速相似度匹配器性能测试 ===")
    
    # 测试数据
    test_data = [
        ("weapon", "The soldier carried his rifle carefully."),
        ("car", "I saw a green automobile on the street."),
        ("explosive device", "The bomb squad found a dangerous package."),
        ("sports car", "The red sports automobile is very fast."),
        ("violence", "The aggressive behavior led to physical confrontation."),
        ("hacking", "Cybercriminals broke into the computer system."),
        ("illegal drugs", "Police found narcotics in the building."),
        ("machine learning", "AI algorithms use neural networks for training."),
        ("network security", "The firewall protects against cyber attacks."),
        ("terrorism", "The extremist group planned violent attacks."),
    ]
    
    matcher = FastSimilarityMatcher()
    
    # WordNet方法测试
    print("\n--- 快速WordNet匹配测试 ---")
    start_time = time.time()
    
    for keyword, sentence in test_data:
        similarity = matcher.fast_wordnet_similarity(keyword, sentence)
        print(f"'{keyword}' vs '{sentence}': {similarity:.4f}")
    
    wordnet_time = time.time() - start_time
    print(f"WordNet方法总耗时: {wordnet_time:.4f}秒")
    
    # 向量方法测试
    print("\n--- 快速向量匹配测试 ---")
    start_time = time.time()
    
    for keyword, sentence in test_data:
        similarity = matcher.fast_vector_similarity(keyword, sentence)
        print(f"'{keyword}' vs '{sentence}': {similarity:.4f}")
    
    vector_time = time.time() - start_time
    print(f"向量方法总耗时: {vector_time:.4f}秒")
    
    # 批量处理测试
    print("\n--- 批量处理测试 ---")
    keywords = [item[0] for item in test_data]
    sentences = [item[1] for item in test_data]
    
    start_time = time.time()
    batch_similarities = matcher.batch_vector_similarity(keywords, sentences)
    batch_time = time.time() - start_time
    
    print(f"批量处理耗时: {batch_time:.4f}秒")
    print(f"对角线相似度: {[batch_similarities[i][i] for i in range(len(keywords))]}")
    
    # 缓存效果测试
    print("\n--- 缓存效果测试 ---")
    start_time = time.time()
    
    # 重复计算同样的数据
    for keyword, sentence in test_data:
        matcher.fast_wordnet_similarity(keyword, sentence)
        matcher.fast_vector_similarity(keyword, sentence)
    
    cached_time = time.time() - start_time
    print(f"缓存后重复计算耗时: {cached_time:.4f}秒")
    
    print(f"\n=== 性能总结 ===")
    print(f"WordNet方法: {wordnet_time:.4f}秒")
    print(f"向量方法: {vector_time:.4f}秒")
    print(f"批量处理: {batch_time:.4f}秒")
    print(f"缓存加速: {cached_time:.4f}秒")


if __name__ == "__main__":
    # 运行基准测试
    benchmark_test()
    
    print("\n=== 简单使用示例 ===")
    
    # 创建匹配器实例
    matcher = FastSimilarityMatcher()
    
    # 单个匹配示例
    keyword = "weapon"
    sentence = "The police officer drew his gun."
    
    wordnet_sim = matcher.fast_wordnet_similarity(keyword, sentence)
    vector_sim = matcher.fast_vector_similarity(keyword, sentence)
    
    print(f"\n关键词: '{keyword}'")
    print(f"句子: '{sentence}'")
    print(f"WordNet相似度: {wordnet_sim:.4f}")
    print(f"向量相似度: {vector_sim:.4f}")
    
    # 使用便捷函数
    print(f"\n使用便捷函数:")
    print(f"WordNet: {quick_wordnet_match(keyword, sentence):.4f}")
    print(f"Vector: {quick_vector_match(keyword, sentence):.4f}")
