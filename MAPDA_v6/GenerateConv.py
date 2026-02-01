import argparse
import Group
import utils
from datetime import datetime
import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

logger = logging.getLogger("mapda")

def run_group(api_pair, 
            role_num, 
            round_num, 
            personality_flag, 
            model, 
            scenario_code, 
            question_folder_path,
            output_dir,
            max_rows_per_file):

    # 处理模型路径，将路径名转为安全的文件名
    base = os.path.basename(model.rstrip("/")) if "/" in model else model
    safe_model_name = re.sub(r"[^A-Za-z0-9_.-]", "_", base)
    saving_file_name = f"{role_num}_{round_num}_{safe_model_name}_{personality_flag}_{scenario_code}.jsonl"
    subdir = os.path.join(output_dir, personality_flag, scenario_code)
    saving_path = os.path.join(subdir, saving_file_name)
    
    os.makedirs(subdir, exist_ok=True)
    if not os.path.exists(saving_path):
        with open(saving_path, 'w') as f:
            pass
            
    try:
        group = Group.Group(
            api_pair=api_pair,
            role_num=role_num,
            round_num=round_num,
            personality_flag=personality_flag,
            model=model,
            scenario_code=scenario_code,
            question_folder_path=question_folder_path,
            max_rows_per_file=max_rows_per_file
        )
        group.run_conversation(saving_path)
        logger.info(f"完成: roles={role_num} rounds={round_num} model={safe_model_name} persona={personality_flag} scene={scenario_code}")
    except Exception as e:
        logger.warning(f"执行出错: roles={role_num} rounds={round_num} model={safe_model_name} -> {e}")

def _parse_list(value: str):
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]

def _normalize_scenario(code: str):
    if not code:
        return code
    code_upper = code.strip().upper()
    if code_upper == "BUSSINESS":
        return "BUSINESS"
    return code_upper

def _build_paras_list(api_pair, args):
    scenario_codes = [_normalize_scenario(c) for c in _parse_list(args.scenarios)]
    if not scenario_codes:
        scenario_codes = ["INTERVIEW", "DEBATE", "COURT", "INVESTIGATION", "BUSINESS", "ACADEMIC", "GAME"]

    personality_flags = _parse_list(args.personalities) or ["mbti"]

    paras_list = []
    for rln in range(args.role_min, args.role_max + 1):
        for rndn in range(args.round_min, args.round_max + 1):
            for personality_flag in personality_flags:
                for scenario_code in scenario_codes:
                    paras = {
                        "api_pair": api_pair,
                        "role_num": rln,
                        "round_num": rndn,
                        "personality_flag": personality_flag,
                        "model": args.model,
                        "scenario_code": scenario_code,
                        "question_folder_path": args.question_folder,
                        "output_dir": args.output_dir,
                        "max_rows_per_file": args.max_rows_per_file,
                    }
                    paras_list.append(paras)
    return paras_list

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-name", default="custom", help="Api.json里的name字段")
    parser.add_argument("--model", default="qwen3-32b", help="模型名称或路径")
    parser.add_argument("--personalities", default="mbti", help="逗号分隔: none,mbti,big5")
    parser.add_argument("--scenarios", default="", help="逗号分隔: INTERVIEW,DEBATE,COURT,INVESTIGATION,BUSINESS,ACADEMIC,GAME")
    parser.add_argument("--question-folder", default="./Data/Chains", help="问题CSV所在目录")
    parser.add_argument("--output-dir", default="./outputs", help="输出jsonl目录")
    parser.add_argument("--max-rows-per-file", type=int, default=0, help="每个类别前N条; 0表示不限制")
    parser.add_argument("--role-min", type=int, default=2)
    parser.add_argument("--role-max", type=int, default=5)
    parser.add_argument("--round-min", type=int, default=3)
    parser.add_argument("--round-max", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=0, help="线程数上限; 0表示使用任务数")
    parser.add_argument("--dry-run", action="store_true", help="只打印任务列表，不执行")
    args = parser.parse_args()

    api_pair = utils.load_api(args.api_name)
    if not api_pair:
        raise ValueError(f"Api.json中找不到name={args.api_name}")

    paras_list = _build_paras_list(api_pair, args)
    if args.dry_run:
        for params in paras_list:
            print(f"plan: roles={params['role_num']} rounds={params['round_num']} "
                  f"persona={params['personality_flag']} scene={params['scenario_code']} "
                  f"model={params['model']}")
        raise SystemExit(0)
    max_workers = args.max_workers if args.max_workers and args.max_workers > 0 else len(paras_list)
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for params in paras_list:
                future = executor.submit(
                    run_group,
                    params["api_pair"],
                    params["role_num"],
                    params["round_num"],
                    params["personality_flag"],
                    params["model"],
                    params["scenario_code"],
                    params["question_folder_path"],
                    params["output_dir"],
                    params["max_rows_per_file"],
                )
                futures.append(future)
            for future in tqdm(futures, total=len(futures), desc="tasks"):
                future.result()
    except KeyboardInterrupt:
        logger.warning("收到中断信号，正在停止任务...")
        # ThreadPoolExecutor 在 with 退出时会等待当前任务，无法强杀线程
        # 通过抛出异常提前结束主循环
        raise
    finally:
        logger.info("所有任务结束")
    
