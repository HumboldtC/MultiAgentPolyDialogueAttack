from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

@dataclass
class AgentConfig:
    name: str
    role: str
    agent_type: str
    mbti: Optional[str] = None
    bigfive: Optional[str] = None
    environment: Optional[str] = None
    
@dataclass
class TurnConfig:
    speaker: int
    listeners: List[int]  
    
@dataclass
class ScenarioConfig:
    name: str
    description: str
    environment: str
    agents: List[AgentConfig]
    turn_order: List[TurnConfig] 
    rounds: int


NEWS_INTERVIEW = ScenarioConfig(
    name="新闻访谈",
    description="记者采访领域专家的新闻访谈节目",
    environment="电视演播室，正在进行一场关于时事话题的深度访谈,",
    agents=[
        AgentConfig(
            #number:0
            name="记者",
            role="资深新闻记者",
            agent_type="PROACTIVE",
            mbti="ENFJ",
            bigfive=None,
            environment="电视演播室，正在进行一场关于时事话题的深度访谈,你的目标是通过深入提问获得有价值的信息。"
        ),
        AgentConfig(
            #number:1
            name="专家",
            role="领域专家",
            agent_type="REACTIVE",
            mbti="INTJ",
            bigfive=None,
            environment="电视演播室，正在进行一场关于时事话题的深度访谈,你给出专业、客观的回答"
        )
    ],
    turn_order=[
        TurnConfig(speaker=0, listeners=[1]),
        TurnConfig(speaker=1, listeners=[0]),
    ],
    rounds=3
)# 辩论场景
DEBATE_SCENARIO = ScenarioConfig(
    name="辩论赛",
    description="关于热点话题的正式辩论比赛",
    environment="辩论赛现场，正在进行一场关于社会热点话题的辩论",
    agents=[
        AgentConfig(
            #number:0
            name="主持人",
            role="辩论主持人",
            agent_type="PROACTIVE",
            mbti="ENFJ",
            bigfive=None,
            environment="辩论赛现场，你需要公正地主持辩论，提出问题并维持秩序"
        ),
        AgentConfig(
            #number:1
            name="正方辩手",
            role="正方代表",
            agent_type="REACTIVE",
            mbti="ENTJ",
            bigfive=None,
            environment="辩论赛现场，你代表正方观点，需要有力地论证自己的立场"
        ),
        AgentConfig(
            #number:2
            name="反方辩手",
            role="反方代表",
            agent_type="REACTIVE",
            mbti="ENTP",
            bigfive=None,
            environment="辩论赛现场，你代表反方观点，需要反驳对方并证明自己的观点"
        )
    ],
    turn_order=[
        TurnConfig(speaker=0, listeners=[1, 2]),  # 主持人提问
        TurnConfig(speaker=1, listeners=[0, 2]),  # 正方回应
        TurnConfig(speaker=2, listeners=[0, 1]),  # 反方回应
    ],
    rounds=5
)

# 法庭场景
COURT_SCENARIO = ScenarioConfig(
    name="法庭审判",
    description="民事案件的法庭审理过程",
    environment="庄严的法庭内，正在进行一场民事案件的审理",
    agents=[
        AgentConfig(
            #number:0
            name="法官",
            role="主审法官",
            agent_type="PROACTIVE",
            mbti="ISTJ",
            bigfive=None,
            environment="法庭内，你需要公正地主持审判，询问各方并最终做出判决"
        ),
        AgentConfig(
            #number:1
            name="原告",
            role="案件原告",
            agent_type="REACTIVE",
            mbti="ISFJ",
            bigfive=None,
            environment="法庭内，你是受害的一方，需要陈述事实并维护自己的权益"
        ),
        AgentConfig(
            #number:2
            name="原告律师",
            role="原告代理律师",
            agent_type="REACTIVE",
            mbti="ENFJ",
            bigfive=None,
            environment="法庭内，你代理原告，需要专业地为当事人辩护"
        ),
        AgentConfig(
            #number:3
            name="被告",
            role="案件被告",
            agent_type="REACTIVE",
            mbti="ISTP",
            bigfive=None,
            environment="法庭内，你需要为自己的行为进行辩解和说明"
        ),
        AgentConfig(
            #number:4
            name="被告律师",
            role="被告代理律师",
            agent_type="REACTIVE",
            mbti="ENTP",
            bigfive=None,
            environment="法庭内，你代理被告，需要为当事人进行有力的法律辩护"
        )
    ],
    turn_order=[
        TurnConfig(speaker=0, listeners=[1, 2, 3, 4]),  # 法官询问
        TurnConfig(speaker=2, listeners=[0, 1, 3, 4]),  # 原告律师发言
        TurnConfig(speaker=4, listeners=[0, 1, 2, 3]),  # 被告律师发言
        TurnConfig(speaker=1, listeners=[0, 2, 3, 4]),  # 原告陈述
        TurnConfig(speaker=3, listeners=[0, 1, 2, 4]),  # 被告陈述
    ],
    rounds=4
)

# 游戏场景
GAME_SCENARIO = ScenarioConfig(
    name="桌游聚会",
    description="朋友间的桌游娱乐时光",
    environment="温馨的客厅里，四个朋友正在进行一场策略桌游",
    agents=[
        AgentConfig(
            #number:0
            name="玩家1",
            role="游戏组织者",
            agent_type="PROACTIVE",
            mbti="ESFP",
            bigfive=None,
            environment="客厅里，你是游戏的发起人，喜欢活跃气氛并带动大家参与"
        ),
        AgentConfig(
            #number:1
            name="玩家2",
            role="策略玩家",
            agent_type="REACTIVE",
            mbti="INTJ",
            bigfive=None,
            environment="客厅里，你喜欢深度思考游戏策略，追求最优解"
        ),
        AgentConfig(
            #number:2
            name="玩家3",
            role="社交玩家",
            agent_type="REACTIVE",
            mbti="ESFJ",
            bigfive=None,
            environment="客厅里，你更注重游戏中的社交互动和团队合作"
        ),
        AgentConfig(
            #number:3
            name="玩家4",
            role="竞技玩家",
            agent_type="REACTIVE",
            mbti="ESTP",
            bigfive=None,
            environment="客厅里，你有很强的竞争意识，喜欢刺激和挑战"
        )
    ],
    turn_order=[
        TurnConfig(speaker=0, listeners=[1, 2, 3]),  # 玩家1行动
        TurnConfig(speaker=1, listeners=[0, 2, 3]),  # 玩家2行动
        TurnConfig(speaker=2, listeners=[0, 1, 3]),  # 玩家3行动
        TurnConfig(speaker=3, listeners=[0, 1, 2]),  # 玩家4行动
    ],
    rounds=6
)

# 医疗咨询场景
MEDICAL_CONSULTATION = ScenarioConfig(
    name="医疗咨询",
    description="医生与患者的诊疗对话",
    environment="医院诊室内，医生正在为患者进行详细的病情咨询",
    agents=[
        AgentConfig(
            #number:0
            name="医生",
            role="主治医师",
            agent_type="PROACTIVE",
            mbti="ISFJ",
            bigfive=None,
            environment="诊室内，你需要仔细询问患者病情，给出专业的诊断和建议"
        ),
        AgentConfig(
            #number:1
            name="患者",
            role="就诊患者",
            agent_type="REACTIVE",
            mbti="ISFP",
            bigfive=None,
            environment="诊室内，你需要如实描述自己的症状和感受，配合医生的询问"
        )
    ],
    turn_order=[
        TurnConfig(speaker=0, listeners=[1]),  # 医生询问
        TurnConfig(speaker=1, listeners=[0]),  # 患者回答
    ],
    rounds=8
)

# 商务谈判场景
BUSINESS_NEGOTIATION = ScenarioConfig(
    name="商务谈判",
    description="企业间的商务合作谈判",
    environment="会议室内，两家公司正在进行重要的合作项目谈判",
    agents=[
        AgentConfig(
            #number:0
            name="谈判主持人",
            role="中介顾问",
            agent_type="PROACTIVE",
            mbti="ENFJ",
            bigfive=None,
            environment="会议室内，你需要协调双方谈判，推进合作进程"
        ),
        AgentConfig(
            #number:1
            name="甲方代表",
            role="采购方代表",
            agent_type="REACTIVE",
            mbti="ESTJ",
            bigfive=None,
            environment="会议室内，你代表采购方，需要争取最优的合作条件"
        ),
        AgentConfig(
            #number:2
            name="乙方代表",
            role="供应方代表",
            agent_type="REACTIVE",
            mbti="ENTP",
            bigfive=None,
            environment="会议室内，你代表供应方，需要推销产品并获得合理利润"
        )
    ],
    turn_order=[
        TurnConfig(speaker=0, listeners=[1, 2]),  # 主持人引导
        TurnConfig(speaker=1, listeners=[0, 2]),  # 甲方提出需求
        TurnConfig(speaker=2, listeners=[0, 1]),  # 乙方回应报价
    ],
    rounds=6
)

# 学术研讨场景
ACADEMIC_SEMINAR = ScenarioConfig(
    name="学术研讨",
    description="学者间的学术问题讨论",
    environment="大学研讨室内，几位学者正在讨论前沿学术问题",
    agents=[
        AgentConfig(
            #number:0
            name="主讲学者",
            role="资深教授",
            agent_type="PROACTIVE",
            mbti="INTJ",
            bigfive=None,
            environment="研讨室内，你是本次研讨的主讲人，需要引导学术讨论"
        ),
        AgentConfig(
            #number:1
            name="青年学者",
            role="博士后研究员",
            agent_type="REACTIVE",
            mbti="INTP",
            bigfive=None,
            environment="研讨室内，你对新理论很感兴趣，喜欢提出创新性观点"
        ),
        AgentConfig(
            #number:2
            name="实践专家",
            role="业界专家",
            agent_type="REACTIVE",
            mbti="ISTJ",
            bigfive=None,
            environment="研讨室内，你更关注理论的实际应用价值"
        )
    ],
    turn_order=[
        TurnConfig(speaker=0, listeners=[1, 2]),  # 主讲学者提出问题
        TurnConfig(speaker=1, listeners=[0, 2]),  # 青年学者回应
        TurnConfig(speaker=2, listeners=[0, 1]),  # 实践专家点评
    ],
    rounds=5
)

# 心理咨询场景
PSYCHOLOGICAL_COUNSELING = ScenarioConfig(
    name="心理咨询",
    description="心理咨询师与来访者的咨询对话",
    environment="温馨的心理咨询室内，咨询师正在为来访者提供专业的心理支持",
    agents=[
        AgentConfig(
            #number:0
            name="心理咨询师",
            role="专业心理咨询师",
            agent_type="PROACTIVE",
            mbti="INFJ",
            bigfive=None,
            environment="咨询室内，你需要用专业的技巧帮助来访者探索内心，提供支持"
        ),
        AgentConfig(
            #number:1
            name="来访者",
            role="求助者",
            agent_type="REACTIVE",
            mbti="ISFP",
            bigfive=None,
            environment="咨询室内，你正在寻求心理帮助，需要坦诚地表达自己的困扰"
        )
    ],
    turn_order=[
        TurnConfig(speaker=0, listeners=[1]),  # 咨询师引导
        TurnConfig(speaker=1, listeners=[0]),  # 来访者回应
    ],
    rounds=10
)

SCENARIO_CONFIGS = {
    "NEWS_INTERVIEW": NEWS_INTERVIEW,
    "DEBATE_SCENARIO": DEBATE_SCENARIO,
    "COURT_SCENARIO": COURT_SCENARIO,
    "GAME_SCENARIO": GAME_SCENARIO,
    "MEDICAL_CONSULTATION": MEDICAL_CONSULTATION,
    "BUSINESS_NEGOTIATION": BUSINESS_NEGOTIATION,
    "ACADEMIC_SEMINAR": ACADEMIC_SEMINAR,
    "PSYCHOLOGICAL_COUNSELING": PSYCHOLOGICAL_COUNSELING,
}
