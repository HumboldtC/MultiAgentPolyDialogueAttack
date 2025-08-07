import Group
import utils
from datetime import datetime
import os
from concurrent.futures import ThreadPoolExecutor

def run_group(api_pair, 
            role_num, 
            round_num, 
            personality_flag, 
            model, 
            scenario_code, 
            question_folder_path):
    """根据传入的参数运行 Group 实例 (适用于线程池)"""

    saving_file_name = f"{role_num}_{round_num}_{model}_{personality_flag}_{scenario_code}.jsonl"
    
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
        print(f"正在运行 Group，参数: {role_num}_{round_num}_{model}_{personality_flag}_{scenario_code}")
        group.run_conversation(saving_file_name)
        print(f"Group 运行完毕，参数: {role_num}_{round_num}_{model}_{personality_flag}_{scenario_code}")
    except Exception as e:
        print(f"运行出错，参数: {role_num}_{round_num}_{model}_{personality_flag}_{scenario_code}, 错误: {e}")


if __name__ == "__main__":
    # 加载API配置
    api_pair = utils.load_api("Intergrated")
    

    paras_list=[]
    for rln in range(2,6):
        for rndn in range(3,6):
            paras={"api_pair":api_pair,
                   "role_num":rln,
                   "round_num":rndn,
                   "personality_flag":"mbti",
                   "model":"claude-sonnet-4-0",
                   "scenario_code":"INTERVIEW",
                   "question_folder_path":"./Data/Chains"}
            paras_list.append(paras)
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
    print("=======FINISHED==========")
    # group = Group.Group(
    #     api_pair=api_pair,
    #     role_num=2,
    #     round_num=3,
    #     personality_flag="mbti",
    #     model="gpt-4",
    #     scenario_code="INTERVIEW",
    #     question_folder_path="./Data/Chains"       
    # )
    # now = datetime.now()
    # formatted_time = now.strftime("%m%d%H%M%S")
    # saving_file_name=f"A{formatted_time}.jsonl"
    # if not os.path.exists(saving_file_name):
    #     with open(saving_file_name, 'w') as f:
    #         pass
    # group.run_conversation(saving_file_name)