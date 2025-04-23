import os
import re
import csv
import json
import polars
import requests
from typing import Tuple


def load_api(api_name: str) -> Tuple[str, str]:
    with open('Api.json', 'r') as file:
        data = json.load(file)
    api_dict = {i['name']: (i['url'], i['key']) for i in data['api']}
    return api_dict.get(api_name)


def send_data(data, api_pair):
    url, api_key = api_pair
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.post(url, headers=headers, json=data)
    result = ""
    if response.status_code == 200:
        result = response.json()
        # print(result)
        result = result['choices'][0]['message']['content']
    else:
        result = "ERROR CODE:" + str(response.status_code) + "\n响应内容:" + str(response.content)
    return result


def init_data(model: str, role: str, content: str, temperature: float = 0.6) -> dict:
    if model=="o3-mini":
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
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        print("ERROR CODE:" + str(response.status_code) + "\n响应内容:" + str(response.content))
        return

def extract_json(text):
    m = re.search(r'(\{.*?\})', text, re.DOTALL)
    if not m:
        raise ValueError("ERROR: Failed to extract JSON")
    return json.loads(m.group(1))

# TODO：全程使用df
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
    print("chain saved")


