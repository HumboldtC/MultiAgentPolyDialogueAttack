import os
import json

clean_data = []
with open("2_3_Qwen3-30B-A3B-Instruct-2507_mbti_INTERVIEW.jsonl", "r") as f:
    lines = f.readlines()
    for each in lines:
        if "Connection error." not in each:
            clean_data.append(json.loads(each))

with open("2_3_Qwen3-30B-A3B-Instruct-2507_mbti_INTERVIEW_clean.jsonl", "w") as f:
    for each in clean_data:
        f.write(json.dumps(each, ensure_ascii=False) + "\n")