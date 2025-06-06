from utils import send_prompt, init_data, add_data, chain_system_prompt_cn, load_api
import re
import json
from abc import abstractmethod
import asyncio
import polars as pl

class BaseAgent:
    def __init__(self,
                 name: str,
                 role: str,
                 model: str,
                 mbti: str,
                 bigfive: str,
                 environment: str,
                 api_pair,
                 ):
        # 初始设定值
        self.model = model
        self.name = name
        self.role = role
        self.mbti = mbti
        self.bigfive = bigfive
        self.api_pair = api_pair
        self.env = environment

        # 异步生成所有描述
        asyncio.run(self._initialize_descriptions())

        # 存储功能
        self.recent_received_message = []
        self.message_data = self.generate_personality_backstory()

    async def _initialize_descriptions(self):
        # 创建异步任务
        tasks = [
            self.generate_role_description_async(),
            self.generate_mbti_description_async(),
            self.generate_bigfive_description_async()
        ]

        # 等待所有任务完成
        results = await asyncio.gather(*tasks)

        # 分配结果
        self.role_description = results[0]
        self.mbti_description = results[1]
        self.bigfive_description = results[2]

    async def generate_role_description_async(self):
        return await asyncio.to_thread(self.generate_role_description)

    async def generate_mbti_description_async(self):
        return await asyncio.to_thread(self.generate_mbti_description)

    async def generate_bigfive_description_async(self):
        return await asyncio.to_thread(self.generate_bigfive_description)

    def generate_role_description(self):
        if self.mbti == None:
            return None
        else:
            prompt = "请为以下职业生成一个角色描述和特点。描述应该包括有且仅有五个关键特征，简明扼要，突出角色的特点、工作方式、专业能力及对社会和道德的态度，要包括一个职业独特于其他职业的负面评价作为特点。每个特征用大括号包括，不需要进一步描述。"
            prompt += "职业："
            prompt += self.role
            response = send_prompt(init_data(self.model, "user", prompt), self.api_pair)
            pattern = r'\{(.*?)\}'
            description = re.findall(pattern, response)
            return description

    def generate_mbti_description(self):
        if self.mbti == None:
            return None
        else:
            prompt = "请生成下面MBTI人格类型的详细描述，描述应该包括有且仅有5个关键特征，简明扼要，突出他们的典型性格特点、行为方式、优点、缺点。每个特征用大括号包括，不需要进一步描述。"
            prompt += "MBTI："
            prompt += self.mbti
            response = send_prompt(init_data(self.model, "user", prompt), self.api_pair)
            pattern = r'\{(.*?)\}'
            description = re.findall(pattern, response)
            return description

    def generate_bigfive_description(self):
        if self.bigfive == None:
            return None
        else:
            prompt = "描述下面Big Five人格维度的关键特征，包括高分者和低分者的典型性格表现、优点、缺点、行为倾向，以及在不同情境下的表现。简明扼要，每个特征用大括号包括，不需要进一步描述。"
            prompt += "Big Five 人格维度："
            prompt += self.bigfive
            response = send_prompt(init_data(self.model, "user", prompt), self.api_pair)
            pattern = r'\{(.*?)\}'
            description = re.findall(pattern, response)
            return description

    def generate_personality_backstory(self):
        backstory = "你是" + self.role + "，你有如下特质："
        for i in self.role_description:
            backstory += i + ","
        backstory += "。"
        if self.mbti != None:
            backstory += "你的MBTI是" + self.mbti + "，你有如下特质："
            for i in self.mbti_description:
                backstory += i + ","
            backstory += "。"
        if self.bigfive != None:
            backstory += "你的大五型人格是" + self.bigfive + "，你有如下特质："
            for i in self.bigfive_description:
                backstory += i + ","
            backstory += "。"
        if self.env != None:
            backstory += "你现在所处的场景是" + self.env + "。"
        backstory += "根据你所处的环境，"
        if self.role != None:
            backstory += "结合你的角色"
        if self.mbti != None and self.bigfive != None:
            backstory += "和性格特质"
        backstory += "，下面将会有一些任务给你，请尽可能完成的贴合要求且效果出色。"
        return init_data(self.model, "system", backstory)

    def receive_message(self, message_list: list):
        # 有发就要有收
        matched_dicts = []
        for item in message_list:
            if self.name in item.get('to'):
                matched_dicts.append(item)
        self.recent_received_message.append(matched_dicts[-1])

    def send_message(self, to, content):
        # to all的话发N遍不要写all
        message = {
            'from': self.name,
            'to': to,
            'content': content
        }
        return message


class ProactiveAgent(BaseAgent):
    def __init__(self,
                 name: str,
                 role: str,
                 model: str,
                 mbti: str,
                 bigfive: str,
                 environment: str,
                 file:str,
                 row:int,
                 api_pair
                 ):
        super().__init__(name, role, model, mbti, bigfive, environment,api_pair)
        self.question_chain=self.read_question_chain(file,row)
    def read_question_chain(self, file_path: str,row:int) -> List[str]:
        
        try:
            df = pl.read_csv(file_path)
            row_list = df.row(row)
            
            # 转换为字符串列表，保留所有单元格（包括空值）
            result = [str(cell) if cell is not None else "" for cell in row_list]
            return result
            
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return []

class ReactiveAgent(BaseAgent):
    def __init__(self,
                 name: str,
                 role: str,
                 model: str,
                 mbti: str,
                 bigfive: str,
                 environment: str,
                 api_pair
                 ):
        super().__init__(name, role, model, mbti, bigfive,  environment,api_pair)

    def get_response(self):
        content = ""
        for m in self.recent_received_message:
            tmp = m["from"] + "说：" + m["content"]
            content += tmp
        self.message_data = add_data(self.message_data, "user", content)
        response = send_prompt(self.message_data, self.api_pair)
        self.message_data = add_data(self.message_data, "assistant", response)
        self.recent_received_message = []
        return response
