import os
import re
import csv
import json
import logging
import threading
import time
from openai import OpenAI
from typing import Tuple

# 确保不走代理，避免 httpx 走 SOCKS/HTTP 代理
for _key in (
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
    "all_proxy", "ALL_PROXY",
):
    os.environ.pop(_key, None)

# logger: default to WARNING; enable DEBUG by setting MAPDA_LOG_LEVEL=DEBUG
logger = logging.getLogger("mapda")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    level_name = os.environ.get("MAPDA_LOG_LEVEL", "WARNING").upper()
    logger.setLevel(getattr(logging, level_name, logging.WARNING))

# Rate limiting (seconds between requests across threads)
_min_interval = float(os.environ.get("MAPDA_MIN_REQUEST_INTERVAL", "0") or "0")
_rate_lock = threading.Lock()
_last_request_ts = 0.0

def _throttle():
    global _last_request_ts
    if _min_interval <= 0:
        return
    with _rate_lock:
        now = time.monotonic()
        wait = _min_interval - (now - _last_request_ts)
        if wait > 0:
            time.sleep(wait)
        _last_request_ts = time.monotonic()


def load_api(api_name: str) -> Tuple[str, str]:
    with open('Api.json', 'r') as file:
        data = json.load(file)
    api_dict = {i['name']: (i['url'], i['key']) for i in data['api']}
    return api_dict.get(api_name)


def send_data(data, api_pair):
    url, api_key = api_pair
    
    try:
        _throttle()
        logger.debug(f"发送请求到: {url}")
        logger.debug(f"请求数据(隐藏messages长度): model={data.get('model')} messages_len={len(data.get('messages', []))}")
        
        # 创建OpenAI客户端
        client = OpenAI(base_url=url, api_key="EMPTY" if api_key == "None" else api_key, timeout=60)
        
        # 提取消息和模型信息
        model = data.get("model", "")
        messages = data.get("messages", [])
        temperature = data.get("temperature", 0.6)
        extra_body = {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        max_tokens = int(os.environ.get("MAPDA_MAX_TOKENS", "1024") or "1024")
        
        logger.debug(f"使用模型: {model}")
        logger.debug(f"消息条数: {len(messages)}")
        
        # 调用OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            extra_body=extra_body,
            max_tokens=max_tokens,
        )
        
        logger.debug("响应成功")
        
        # 提取响应内容
        content = response.choices[0].message.content
        # 不打印生成内容到终端
        return content
            
    except Exception as e:
        error_msg = f"API调用异常: {str(e)}"
        logger.warning(error_msg)
        return error_msg


def init_data(model: str, role: str, content: str, temperature: float = 0.6) -> dict:
    if role == "system":
        max_tokens = os.environ.get("MAPDA_MAX_TOKENS", "1024")
        content = (
            f"{content}\n\n"
            f"Please keep your reply within {max_tokens} tokens."
        )
    if model=="o3-mini" or model=="o1-mini" or model=="gpt-4.1":
        data = {
            "model": model,
            "messages": [{"role": role,
                          "content": content}],
        }
    else:
        data = {
            "model": model,
            "messages": [{"role": role,
                          "content": content}],
            "temperature": temperature
        }
    return data


def add_data(ori_data, role: str, content: str):
    ori_data["messages"].append({
        "role": role,
        "content": content
    })
    return ori_data




def send_score_data(data, api_pair):
    url, api_key = api_pair
    
    try:
        _throttle()
        # 创建OpenAI客户端
        client = OpenAI(base_url=url, api_key="EMPTY" if api_key == "None" else api_key, timeout=60)
        
        # 提取消息和模型信息
        model = data.get("model", "")
        messages = data.get("messages", [])
        temperature = data.get("temperature", 0.6)
        extra_body = {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        max_tokens = int(os.environ.get("MAPDA_MAX_TOKENS", "1024") or "1024")
        
        # 调用OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            extra_body=extra_body,
            max_tokens=max_tokens,
        )
        
        # 返回完整的响应对象转换为字典
        return {
            "choices": [
                {
                    "message": {
                        "content": response.choices[0].message.content
                    }
                }
            ]
        }
        
    except Exception as e:
        logger.warning(f"API调用异常: {str(e)}")
        return None

def extract_json(text):
    m = re.search(r'(\{.*?\})', text, re.DOTALL)
    if not m:
        raise ValueError("ERROR: Failed to extract JSON")
    return json.loads(m.group(1))


def append_list_to_csv(file_path, data_list, sbj: str):
    file_exists = os.path.isfile(file_path)
    # 创建一个新的列表，将sbj放在第一位
    new_data_list = [sbj] + data_list

    # 如果文件不存在，需要创建并添加BOM
    if not file_exists:
        with open(file_path, 'w', newline='', encoding='utf-8-sig') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(new_data_list)
    else:
        # 如果文件已存在，直接追加
        with open(file_path, 'a', newline='', encoding='utf-8-sig') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(new_data_list)
    logger.debug("chain saved")
