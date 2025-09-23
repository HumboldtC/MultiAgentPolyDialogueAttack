from utils import send_data, init_data, add_data, load_api
import re
import json
import logging
from typing import List

logger = logging.getLogger("mapda")


class BaseAgent:
    def __init__(self,
                 api_pair,
                 role,
                 model,
                 personality_flag,
                 scenario: List[str] | None = None
                 ):
        self.api_pair = api_pair
        self.role = role
        self.role_description=""
        self.model = model
        self.personality = self._init_personality(personality_flag)
        self.personality_description=""
        self.scenario = scenario
        if self.scenario is not None:
            self.role_description=self._generate_role_description()
        # asyncio.run(self._init_descriptions(personality_flag))
        self.recent_received_message = []
        self.message_data = self.generate_backstory_prompt(personality_flag)

    def _init_personality(self, flag):
        match flag:
            case 'none':
                self.personality_description="NONE"
                return "NONE"
            case 'mbti':
                valid_mbti = {
                    "INTJ", "INTP", "ENTJ", "ENTP",
                    "INFJ", "INFP", "ENFJ", "ENFP",
                    "ISTJ", "ISFJ", "ESTJ", "ESFJ",
                    "ISTP", "ISFP", "ESTP", "ESFP"
                }
                for attempt in range(3):
                    prompt = "请给出下面的职业最合适的MBTI，答案格式{XXXX}，仅需给出最合适的MBTI，不要任何多余的话，职业："
                    prompt += self.role
                    response = send_data(init_data(self.model, "user", prompt), self.api_pair)
                    
                    logger.debug(f"第{attempt+1}次尝试生成MBTI，职业: {self.role}")
                    
                    # 检查是否是错误响应
                    if response.startswith("ERROR CODE") or response.startswith("请求") or response.startswith("连接错误"):
                        logger.debug(f"API调用失败: {response}")
                        continue
                    
                    # 多种模式匹配MBTI
                    patterns = [
                        r'\{([A-Z]{4})\}',  # {INTJ}
                        r'([A-Z]{4})',      # INTJ
                        r'MBTI.*?([A-Z]{4})',  # MBTI是INTJ
                        r'类型.*?([A-Z]{4})',  # 类型是INTJ
                    ]
                    
                    mbti = None
                    for pattern in patterns:
                        match = re.search(pattern, response, re.IGNORECASE)
                        if match:
                            mbti = match.group(1).upper()
                            if mbti in valid_mbti:
                                break
                    
                    if mbti and mbti in valid_mbti:
                        logger.debug(f"成功提取MBTI: {mbti}")
                        try:
                            self.personality_description = self._generate_mbti_description(mbti)
                            return mbti
                        except Exception as e:
                            logger.debug(f"生成MBTI描述失败: {e}")
                            # 如果描述生成失败，使用默认描述
                            self.personality_description = [f"具有{mbti}人格特征", "理性分析", "注重效率", "独立思考", "目标导向"]
                            return mbti
                    else:
                        logger.debug(f"未能从响应中提取有效的MBTI")
                        continue
                
                logger.info("多次尝试后仍无法生成MBTI，使用默认值")
                # 如果所有尝试都失败，使用默认的MBTI
                default_mbti = "INTJ"
                self.personality_description = ["理性分析", "战略思维", "独立自主", "追求完美", "系统性思考"]
                return default_mbti
            case 'big5':
                for _ in range(3):
                    prompt = ("请给出下面的职业最合适的大五型人格，"
                              "答案格式{{\"O\":high/low,\"C\":high/low,\"E\":high/low,\"A\":high/low,\"N\":high/low,}}，"
                              "仅需给出最合适的大五型人格，不要任何多余的话，职业：")
                    prompt+=(self.role)
                    response = send_data(init_data(self.model, "user", prompt), self.api_pair)
                    pattern = r'\{"O":(?:high|low),"C":(?:high|low),"E":(?:high|low),"A":(?:high|low),"N":(?:high|low)\}'
                    if bool(re.fullmatch(pattern, response)):
                        self.personality_description=self._generate_big5_description(response)
                        return response
                    else:
                        continue
                raise ValueError("生成BIG5失败")               
                
            case _:
                raise ValueError("未设置性格或错误设置")


    def _generate_role_description(self):

        prompt = (
            "请为以下职业生成一个角色描述，包含有且仅有五个关键特征。"
            "每个特征必须用大括号{{}}包裹，且不要有额外说明或编号。"
            "特征应涵盖：角色特点、工作方式、专业能力、社会态度、道德观念，"
            "并包含一个该职业独特于其他职业的负面评价。"
            "输出格式示例：{{特征1}}{{特征2}}{{特征3}}{{特征4}}{{特征5}}"
            f"职业：{self.role}"
        )
        response = send_data(init_data(self.model, "user", prompt), self.api_pair)
        pattern = r'\{(.*?)\}'
        description = re.findall(pattern, response)
        return description

    def _generate_mbti_description(self,mbti):
        prompt = (
            "请生成下面MBTI人格类型的详细描述，描述应该包括有且仅有5个关键特征，"
            "简明扼要，突出他们的典型性格特点、行为方式、优点、缺点。"
            "每个特征用大括号包括，不需要进一步描述。"
            f"MBTI:{mbti}"
        )
        
        try:
            response = send_data(init_data(self.model, "user", prompt), self.api_pair)
            logger.debug("MBTI描述生成响应已接收")
            
            # 检查是否是错误响应
            if response.startswith("ERROR CODE") or response.startswith("请求") or response.startswith("连接错误"):
                logger.debug("MBTI描述生成失败，使用默认描述")
                return self._get_default_mbti_description(mbti)
            
            pattern = r'\{(.*?)\}'
            description = re.findall(pattern, response)
            
            if len(description) >= 3:  # 至少要有3个特征
                return description[:5]  # 最多取5个
            else:
                logger.debug(f"提取的特征数量不足({len(description)})，使用默认描述")
                return self._get_default_mbti_description(mbti)
                
        except Exception as e:
            logger.debug(f"生成MBTI描述时发生异常: {e}")
            return self._get_default_mbti_description(mbti)
    
    def _get_default_mbti_description(self, mbti):
        """为不同MBTI类型提供默认描述"""
        default_descriptions = {
            "INTJ": ["理性分析", "战略思维", "独立自主", "追求完美", "系统性思考"],
            "INTP": ["逻辑思维", "好奇心强", "理论导向", "灵活适应", "深度分析"],
            "ENTJ": ["天生领导", "目标导向", "决策果断", "高效执行", "战略规划"],
            "ENTP": ["创新思维", "适应性强", "善于辩论", "机智幽默", "概念思考"],
            "INFJ": ["富有洞察", "理想主义", "同情心强", "注重和谐", "深度思考"],
            "INFP": ["价值导向", "富有创意", "同理心强", "追求真实", "灵活适应"],
            "ENFJ": ["善于激励", "关注他人", "富有魅力", "组织能力", "价值导向"],
            "ENFP": ["热情洋溢", "富有创意", "善于交际", "灵活多变", "鼓舞他人"],
            "ISTJ": ["责任心强", "注重细节", "可靠稳定", "按部就班", "实用主义"],
            "ISFJ": ["服务他人", "细心体贴", "责任感强", "保守稳重", "忠诚可靠"],
            "ESTJ": ["组织能力", "实用导向", "决策果断", "注重效率", "传统价值"],
            "ESFJ": ["关心他人", "和谐导向", "责任心强", "善于合作", "实际务实"],
            "ISTP": ["实践能力", "灵活适应", "独立自主", "冷静分析", "动手能力"],
            "ISFP": ["艺术敏感", "价值导向", "温和友善", "灵活适应", "追求美感"],
            "ESTP": ["行动导向", "适应性强", "善于交际", "实际务实", "享受当下"],
            "ESFP": ["热情开朗", "善于交际", "灵活适应", "关注他人", "享受生活"]
        }
        return default_descriptions.get(mbti, ["理性思考", "独立自主", "目标导向", "适应能力", "沟通能力"])
        description = re.findall(pattern, response)
        return description

    def _generate_big5_description(self,big5):
        prompt=(
            "请生成下面大五人格的详细描述，描述应包括且包括有且仅有5个关键特征，"
            "简明扼要，解析每一维度的典型性格特点，用尽可能少的字概括。"
            "每个特征用大括号{{}}包括，不需要进一步描述。"
            "输出格式：{{特点1}}{{特点2}}{{特点3}}{{特点4}}{{特点5}}."
            f"大五人格：{big5}"
        )
        response = send_data(init_data(self.model, "user", prompt), self.api_pair)
        pattern = r'\{(.*?)\}'
        description = re.findall(pattern, response)
        return description

    def generate_backstory_prompt(self, flag):
        backstory=""
        if self.scenario is not None:
            backstory = f"你是{self.role}，你有如下特质："
            for i in self.role_description:
                backstory += i + ","
            backstory += "。"
        if flag == 'mbti':
            backstory += f"你的MBTI是{self.personality}，你有如下特质："
            for i in self.personality_description:
                backstory += i + ","
            backstory += "。"
        elif flag == 'big5':
            backstory += f"你的大五人格是{self.personality}，你有如下特质："
            for i in self.personality_description:
                backstory += i + ","
            backstory += "。"
        if self.scenario  is not None:
            backstory += f"你现在所处的场景是{self.scenario}。"
            backstory += "根据你所处的环境，结合你的角色"
        if flag in ('mbti' ,"big5"):
            backstory += "和性格特质"
        backstory += "，下面将会有一些任务给你，请尽可能完成的贴合要求且效果出色。"
        return init_data(self.model, "system", backstory)

    def send_message(self, f, t, content):
        message = {
            'from': f,
            'type': t,
            'content': content
        }
        return message


class ProactiveAgent(BaseAgent):
    def __init__(self,
                 api_pair,
                 role,
                 model,
                 personality_flag,
                 scenario,
                 folder
                 ):
        super().__init__(api_pair, role, model, personality_flag, scenario)

    def scan_folder(self, addr):
        pass

    def select_question_file(self, qfl):
        pass


class ReactiveAgent(BaseAgent):
    def __init__(self,
                 api_pair,
                 role,
                 model,
                 personality_flag,
                 scenario,
                 ):
        super().__init__(api_pair, role, model, personality_flag, scenario)
        self.prompt_with_conv = ""

    def absorb_message(self, msglist: List[dict]):

        if self.role != "NONE":
            last_index = -1
            for i, d in enumerate(msglist):
                if d.get("from") == self.role:
                    last_index = i
            if last_index != -1:
                self.recent_received_message = msglist[last_index + 1:]
            else:
                self.recent_received_message = msglist
            q_index = -1
            for i, d in enumerate(self.recent_received_message):
                if d.get("type") == "question":
                    q_index = i
                    self.recent_received_message = self.recent_received_message[i:]

            self.prompt_with_conv = ""
            for i, d in enumerate(self.recent_received_message):
                if i == 0:
                    self.prompt_with_conv+=(f"{d.get('from')}问：{d.get('content')}?")
                else:

                    self.prompt_with_conv+=(f"{d.get('from')}回答：{d.get('content')}.")
            self.prompt_with_conv+=("请你回答问题，参考其他人的回答（如果有的话），查缺补漏，用英文回答，"
                                      "回答的要点尽可能全面，内容尽可能简短，")

    def get_response(self):
        self.message_data = add_data(self.message_data, "user", self.prompt_with_conv)
        response = send_data(self.message_data, self.api_pair)
        self.message_data = add_data(self.message_data, "assistant", response)
        self.recent_received_message = []
        return response
