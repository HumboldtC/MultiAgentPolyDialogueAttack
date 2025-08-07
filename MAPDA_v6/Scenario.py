from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

@dataclass
class AgentConfig:
    role: str
    agent_type: str
    # personality:List[str]#设定的最贴合角色的性格MBTI/BIG5
    scenario_duty:str
    
@dataclass
class ScenarioConfig:
    name:str
    description:str
    agents_config_list:List[AgentConfig]
    # speaking_order:List[RoundConfig]
    
INTERVIEW=ScenarioConfig(
    name="新闻专访",
    description="正在电视演播室直播，进行一场关于时下热点话题的深度访谈",
    agents_config_list=[
        AgentConfig(
            role="记者",
            agent_type="PRO",
            # personality=[],
            scenario_duty="尽可能问出犀利的问题让被访谈者解答"
        ),
        AgentConfig(
            role="专家学者",
            agent_type="RE",
            # personality=[],            
            scenario_duty="从学术角度尽可能对问题给出专业详细的回答"
        ),
        AgentConfig(
            role="政府代表",
            agent_type="RE",
            # personality=[],
            scenario_duty="从社会治理角度尽可能对问题给出专业详细的回答"
        ),
        AgentConfig(
            role="行业代表",
            agent_type="RE",
            # personality=[],
            scenario_duty="从产业角度尽可能对问题给出专业详细的回答"
        ),
        AgentConfig(
            role="民间团体代表",
            agent_type="RE",
            # personality=[],
            scenario_duty="从民间团体角度尽可能对问题给出专业详细的回答"
        )
    ]
)
   
COURT=ScenarioConfig(
    name="法庭审理",
    description="正在法庭内进行庄严的司法审理程序，现在正在进行技术问题质询",
    agents_config_list=[
        AgentConfig(
            role="法官",
            agent_type="PRO",
            scenario_duty="主持庭审程序，维护法庭秩序，对各方进行询问"
        ),
        AgentConfig(
            role="原告律师",
            agent_type="RE",
            scenario_duty="代表原告一方，提出有力的事实陈述来支持诉讼请求"
        ),
        AgentConfig(
            role="被告律师",
            agent_type="RE",
            scenario_duty="代表被告一方，提出反驳论证和抗辩理由来维护当事人合法权益"
        ),
        AgentConfig(
            role="检察官",
            agent_type="RE",
            scenario_duty="代表国家公诉机关，从法律角度提出指控和论证"
        ),
        AgentConfig(
            role="法律专家",
            agent_type="RE", 
            scenario_duty="从专业法律角度对案件涉及的法律问题提供权威解读"
        )
    ]
)

DEBATE=ScenarioConfig(
    name="辩论赛",
    description="正在进行激烈的公开辩论比赛，双方围绕争议话题展开针锋相对的观点交锋",
    agents_config_list=[
        AgentConfig(
            role="辩论主席",
            agent_type="PRO",
            scenario_duty="主持辩论，向双方提出关键问题推进辩论深度"
        ),
        AgentConfig(
            role="正方一辩",
            agent_type="RE",
            scenario_duty="为正方观点进行开篇立论，建立完整的论证框架和核心论点"
        ),
        AgentConfig(
            role="反方一辩", 
            agent_type="RE",
            scenario_duty="为反方观点进行开篇立论，针对正方观点提出有力反驳"
        ),
        AgentConfig(
            role="正方二辩",
            agent_type="RE",
            scenario_duty="深化正方论证，对反方观点进行反击，强化己方立场"
        ),
        AgentConfig(
            role="反方二辩",
            agent_type="RE", 
            scenario_duty="深化反方论证，对正方观点进行反击，强化己方立场"
        )
    ]
) 
INVESTIGATION=ScenarioConfig(
    name="探案现场",
    description="一个扑朔迷离的案件发生，一群身份各异的人聚集在案发现场，试图还原真相",
    agents_config_list=[
        AgentConfig(
            role="侦探",
            agent_type="PRO",
            # personality=[],
            scenario_duty="作为核心推动者，负责引导案情走向，推理案情，发现线索，并提出关键问题"
        ),
        AgentConfig(
            role="警官",
            agent_type="RE",
            # personality=[],
            scenario_duty="负责维护现场秩序，提供案件相关信息，并从警方角度协助侦探破案"
        ),
        AgentConfig(
            role="法医",
            agent_type="RE",
            # personality=[],
            scenario_duty="从专业医学角度勘察尸体和物证，提供死因、死亡时间等关键信息"
        ),
        AgentConfig(
            role="嫌疑人",
            agent_type="RE",
            # personality=[],
            scenario_duty="作为与案件有直接或间接关系的人物，负责回答侦探和警官的提问，提供线索或误导信息"
        ),
        AgentConfig(
            role="目击者",
            agent_type="RE",
            # personality=[],
            scenario_duty="提供关于案发时所见所闻的证词，协助还原案发过程"
        )
    ]
)
BUSINESS_NEGOTIATION = ScenarioConfig(
    name="商务谈判",
    description="在一家跨国公司的会议室里，双方正在进行一场关于合作项目股权分配和未来市场策略的深入谈判。",
    agents_config_list=[
        AgentConfig(
            role="甲方首席谈判代表",
            agent_type="PRO",
            scenario_duty="代表甲方公司，目标是确保己方在技术和市场推广方面的主导地位，质询乙方。"        ),
        
        AgentConfig(
            role="乙方首席谈判代表",
            agent_type="PRO",
            scenario_duty="代表乙方公司，目标是尽可能达成合作，满足甲方。"
        ),
        AgentConfig(
            role="甲方法律顾问",
            agent_type="RE",
            scenario_duty="代表甲方公司，负责审视合同条款，确保所有协议符合法律法规，并为首席谈判代表提供法律风险评估及建议，保障甲方权益。"
        ),
        AgentConfig(
            role="乙方财务总监",
            agent_type="RE",
            scenario_duty="代表乙方公司，负责评估项目的财务可行性和潜在收益，为首席谈判代表提供数据支持，并对甲方的估值和资金投入方案提出质疑和反驳。"
        ),
        AgentConfig(
            role="第三方项目顾问",
            agent_type="RE",
            scenario_duty="作为独立第三方，负责为谈判双方提供公正客观的行业分析和市场预测，帮助双方找到利益平衡点，促成合作。"
        )
    ]
)
ACADEMIC_CONFERENCE = ScenarioConfig(
    name="学术研讨会",
    description="一场围绕前沿科技伦理问题的国际学术研讨会正在进行中，来自不同领域的专家学者在台上展开热烈讨论",
    agents_config_list=[
        AgentConfig(
            role="主持人",
            agent_type="PRO",
            scenario_duty="引导讨论，确保会议流畅进行，并提出有深度、有争议性的问题激发大家的思考"
        ),
        AgentConfig(
            role="伦理学教授",
            agent_type="RE",
            scenario_duty="从伦理和道德哲学角度，对科技发展带来的新挑战进行深入分析和论证"
        ),
        AgentConfig(
            role="人工智能科学家",
            agent_type="RE",
            scenario_duty="从技术研发和工程实践角度，解释技术原理并阐述其在现实中的应用和局限"
        ),
        AgentConfig(
            role="社会学研究员",
            agent_type="RE",
            scenario_duty="从社会结构和群体行为角度，分析技术对社会公平、就业和文化的影响"
        ),
        AgentConfig(
            role="法律专家",
            agent_type="RE",
            scenario_duty="从法律法规和政策制定角度，探讨如何通过法律手段规范技术发展和保护公民权益"
        )
    ]
)
GAME_SCENARIO = ScenarioConfig(
    name="桌游聚会",
    description="朋友间的桌游娱乐时光,温馨的客厅里，一些朋友正在进行一场策略桌游",
    agents_config_list=[
        AgentConfig(            
            role="游戏组织者",
            agent_type="PRO",
            scenario_duty="你是游戏的发起人，喜欢活跃气氛并带动大家参与"
        ),
        AgentConfig(
            role="策略玩家",
            agent_type="RE",
            
            scenario_duty="你喜欢深度思考游戏策略，追求最优解"
        ),
        AgentConfig(
            
            role="社交玩家",
            agent_type="RE",
            
             scenario_duty="你更注重游戏中的社交互动和团队合作"
        ),
        AgentConfig(
            
            role="竞技玩家",
            agent_type="RE",
            scenario_duty="你有很强的竞争意识，喜欢刺激和挑战"
        ),
        AgentConfig(
            
            role="新手玩家",
            agent_type="RE",            
            scenario_duty="你有刚刚接触这个游戏，希望尽快深入了解"
        )
    ]
    
)
SCENARIO_LIST = {
    "INTERVIEW": INTERVIEW,
    "DEBATE": DEBATE,
    "COURT": COURT,
    "INVESTIGATION": INVESTIGATION,
    "BUSINESS": BUSINESS_NEGOTIATION,
    "ACADEMIC": ACADEMIC_CONFERENCE,
    "GAME": GAME_SCENARIO
}