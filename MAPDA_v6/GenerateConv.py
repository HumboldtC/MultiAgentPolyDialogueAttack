import Group
import utils
from datetime import datetime
import os
import re
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("mapda")

def run_group(api_pair, 
            role_num, 
            round_num, 
            personality_flag, 
            model, 
            scenario_code, 
            question_folder_path):

    # 处理模型路径，将路径名转为安全的文件名
    base = os.path.basename(model.rstrip("/")) if "/" in model else model
    safe_model_name = re.sub(r"[^A-Za-z0-9_.-]", "_", base)
    saving_file_name = f"{role_num}_{round_num}_{safe_model_name}_{personality_flag}_{scenario_code}.jsonl"
    
    if not os.path.exists(saving_file_name):
        with open(saving_file_name, 'w') as f:
            pass
            
    try:
        group = Group.Group(
            api_pair=api_pair,
            role_num=role_num,
            round_num=round_num,
            personality_flag=personality_flag,
            model=model,
            scenario_code=scenario_code,
            question_folder_path=question_folder_path
        )
        group.run_conversation(saving_file_name)
        logger.info(f"完成: roles={role_num} rounds={round_num} model={safe_model_name} persona={personality_flag} scene={scenario_code}")
    except Exception as e:
        logger.warning(f"执行出错: roles={role_num} rounds={round_num} model={safe_model_name} -> {e}")

   
if __name__ == "__main__":
    
    #Parameters
    llmapi="qwen3" #name of api configuration to use
    personality_flag="mbti"  #choose none or mbti or big5
    model="/beegfs/g3_hzq/icassp2026/model/Qwen3-30B-A3B-Instruct-2507" #name of llm model to use - 使用实际的模型路径
    scenario_code="INTERVIEW"
    # there're 8 pre-set scenario:
    # "INTERVIEW","DEBATE","COURT","INVESTIGATION",
    # "BUSINESS","ACADEMIC","GAME" or your custom

    '''
    Default configuration runs automatically with 8 threads for optimal performance
    '''
    
    api_pair = utils.load_api(llmapi)    
    paras_list=[]
    for rln in range(2,6):
        for rndn in range(3,6):
            paras={"api_pair":api_pair,
                   "role_num":rln,
                   "round_num":rndn,
                   "personality_flag":"mbti",
                   "model":"/beegfs/g3_hzq/icassp2026/model/Qwen3-30B-A3B-Instruct-2507",  # 使用实际的模型路径
                   "scenario_code":"INTERVIEW",
                   "question_folder_path":"./Data/Chains"}
            paras_list.append(paras)
    try:
        with ThreadPoolExecutor(max_workers=len(paras_list)) as executor:
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
                )
                futures.append(future)
            for future in futures:
                future.result()
    except KeyboardInterrupt:
        logger.warning("收到中断信号，正在停止任务...")
        # ThreadPoolExecutor 在 with 退出时会等待当前任务，无法强杀线程
        # 通过抛出异常提前结束主循环
        raise
    finally:
        logger.info("所有任务结束")
    
