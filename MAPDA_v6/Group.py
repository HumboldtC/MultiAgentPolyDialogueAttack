from Agent import ProactiveAgent, ReactiveAgent
from Scenario import SCENARIO_LIST
from pathlib import Path
from typing import List
import polars as pl
from tqdm import tqdm
import json
import re


class Group:
    def __init__(self,
                 api_pair,
                 role_num: int,
                 round_num: int,
                 personality_flag: str,
                 model: str,
                 scenario_code: str,
                 question_folder_path: str,
                 max_rows_per_file: int = 0
                 #  MAS_flag:str,
                 ):
        self.api_pair = api_pair
        self.role_num = role_num
        self.round_num = round_num
        self.personality_flag = personality_flag
        self.model = model
        self.question_file_list = self._scan_folder(question_folder_path)
        self.message_list: List[dict] = []
        self.agents = []
        self.scenario_code = scenario_code
        self.max_rows_per_file = max_rows_per_file
        if scenario_code != "NONE":
            try:
                self.scenario = SCENARIO_LIST[scenario_code]

            except KeyError:
                raise ValueError(
                    f"Invalid scenario_code. "
                ) from None
            self.scenario_prompt_list = []
            self._load_scenario()
        else:
            self.scenario = None
        self._create_agents()

    @staticmethod
    def _scan_folder(addr) -> List[Path]:
        root = Path(addr)
        result = []
        if not root.exists():
            raise FileNotFoundError(f"路径不存在: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"路径不是文件夹: {root}")

        for item in root.rglob('*.csv'):
            result.append(item)

        return sorted(result)

    def _load_scenario(self):
        agents = self.scenario.agents_config_list
        roles = [agent.role for agent in agents[:self.role_num]]
        roles_str = "，".join(roles)
        agent_env_prompt = (
            f"现在是{self.scenario.name}，你{self.scenario.description}。"
            f"现场共有{self.role_num}人，分别是{roles_str}。"
        )
        self.scenario_prompt_list.append(agent_env_prompt)
        # 加载对应agent的duty
        for agent in agents[:self.role_num]:
            duty_prompt = f"你是{agent.role}，你的任务是{agent.scenario_duty}。"
            self.scenario_prompt_list.append(duty_prompt)

    def _create_agents(self):

        for i in range(self.role_num):
            agt = self.scenario.agents_config_list[i]
            scenario_status: bool = (self.scenario is not None)
            role = agt.role if scenario_status else "NONE"
            agent_type = agt.agent_type if scenario_status else ("PRO" if i == 0 else "RE")
            if scenario_status:
                scenario_combination = [
                    self.scenario_prompt_list[0],
                    self.scenario_prompt_list[i + 1]
                ]
            else:
                scenario_combination = None
            agent_config_pack = [self.api_pair,
                                 role,
                                 self.model,
                                 self.personality_flag,
                                 scenario_combination
                                 ]
            match agent_type:
                case "RE":
                    self.agents.append(ReactiveAgent(*agent_config_pack))
                case "PRO":
                    agent_config_pack.append(self.question_file_list)
                    self.agents.append(ProactiveAgent(*agent_config_pack))
                case _:
                    raise ValueError("Agent类型错误")

    def run_conversation(self, saving_addr):
        selected_question_file_list = [
            f for f in self.question_file_list
            if re.search(r'_' + str(self.round_num) + '_', str(f))
        ]
        for qfile in selected_question_file_list:
            c1 = qfile.parent.name
            match = re.search(r'^(.*?)_\d', qfile.name)
            if match:
                c2 = match.group(1)
            result_info = {"category1": c1, "category2": c2,
                           "round_num": self.round_num,
                           "agent_num": self.role_num,
                           "model": self.model,
                           "personality_flag": self.personality_flag,
                           "scenario": self.scenario_code}
            df = pl.read_csv(qfile,separator=";")
            if self.max_rows_per_file and self.max_rows_per_file > 0:
                df = df.head(self.max_rows_per_file)
            for row in tqdm(df.iter_rows(), total=df.height, desc=f"{c1}/{c2}"):
                questions = row[1:]
                for agt in self.agents:
                    agt.reset_conversation()
                self.message_list = []
                with open(saving_addr, "r", encoding='utf-8') as f:
                    lines = f.readlines()
                exist_flag=False
                for l in lines:
                    if questions[0] in l:
                        exist_flag=True
                if exist_flag:
                    continue                 
                for q in questions:
                    role_name = self.agents[0].role if self.agents[0].scenario else "0"
                    self.message_list.append(
                        self.agents[0].send_message(role_name, "question", q))
                    for i in range(1, self.role_num):
                        agt = self.agents[i]
                        agt.absorb_message(self.message_list)
                        r = agt.get_response()
                        r_role_name = agt.role if agt.scenario else str(i)
                        self.message_list.append(
                            agt.send_message(r_role_name, "answer", r))
                if any("API调用异常" in m.get("content", "") for m in self.message_list):
                    continue
                result = result_info.copy()
                result.update(self.msglist_format_transfer(self.message_list))
                result_line = json.dumps(result, ensure_ascii=False)
                with open(saving_addr, "a", encoding='utf-8') as f:
                    f.write(result_line)
                    f.write('\n')

    def msglist_format_transfer(self, msglist):
        roles = []
        for i in range(self.role_num):
            roles.append(msglist[i]["from"])
        rounds = []
        current_round = {}
        for msg in msglist:
            msg_from = msg["from"]
            msg_type = msg["type"]
            msg_content = msg["content"]
            if msg_type == "question":
                if current_round:
                    rounds.append(current_round)
                current_round = {"question": msg_content}
            elif msg_type == "answer":
                answer_key_prefix = "answer"
                answer_count = len([k for k in current_round if k.startswith(answer_key_prefix)]) + 1
                current_round[f"{answer_key_prefix}{answer_count}"] = msg_content
        if current_round:
            rounds.append(current_round)

        return {
            "roles": roles,
            "rounds": rounds
        }
