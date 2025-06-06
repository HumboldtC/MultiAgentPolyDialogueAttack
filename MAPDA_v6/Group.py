from Agent import ProactiveAgent, ReactiveAgent
from Scenario import SCENARIO_CONFIGS


class Group:
    def __init__(self, scenario_name, model, api_pair):
        try:
            self.scenario = SCENARIO_CONFIGS[scenario_name]
        except KeyError:
            # 键不存在时的处理
            available_keys = list(SCENARIO_CONFIGS.keys())
            raise ValueError(
                f"Invalid scenario_name '{scenario_name}'. "
                f"Available options are: {available_keys}"
            ) from None
        self.model = model
        self.api_pair = api_pair
        self.message_list = []
        self.agent_list=self.create_agents()

    def create_agents(self):
        agent_list = []
        for i, agent_config in enumerate(self.scenario.agents):
            if agent_config.agent_type == "PROACTIVE":
                agent = ProactiveAgent(
                    name=agent_config.name,
                    role=agent_config.role,
                    model=self.model,
                    mbti=agent_config.mbti,
                    bigfive=agent_config.bigfive,
                    environment=agent_config.envset,
                    api_pair=self.api_pair
                )
            else:
                agent = ReactiveAgent(
                    name=agent_config.name,
                    role=agent_config.role,
                    model=self.model,
                    mbti=agent_config.mbti,
                    bigfive=agent_config.bigfive,
                    environment=agent_config.envset,
                    api_pair=self.api_pair
                )
            agent_list.append(agent)
        return agent_list
    # def find_agent_by_name(self, name: str):
    #     for ag in self.agent_list:
    #         if ag.name == name:
    #             return ag
    #     raise ValueError(f"No agent named {name}")
    # def execute_workflow(self):
    #     for i in self.scenario.rounds:
    #         for r in self.scenario.turn_order:
    #             if agent_list[r.speaker].agent_type=="PROACTIVE":
    #                 self.message_list.append(
    #                 self.agent_list[r.speaker].send_message(
    #                     for i in r.listeners,
    #                     self.agent_list[r.speaker].read_question_chain()[i]))
    #             else:
    #                 agent_list[r.speaker].receive_message()
    #                 self.message_list.append(
    #                 self.agent_list[r.speaker].send_message(
    #                     for i in r.listeners,
    #                     self.agent_list[r.speaker].get_response()))
    def execute_workflow(self):
        for round_num in range(self.scenario.rounds):
            # print(f"=== Round {round_num + 1} ===")
            
            for turn in self.scenario.turn_order:
                speaker_agent = self.agent_list[turn.speaker]
                listener_agents = [self.agent_list[i] for i in turn.listeners]
                
                #print(f"{speaker_agent.name} is speaking to {[agent.name for agent in listener_agents]}")
                
                if speaker_agent.agent_type == "PROACTIVE":
                    
                    content = speaker_agent.read_question_chain()[round_num]
                    
                    message = speaker_agent.send_message(
                        to=[agent.name for agent in listener_agents],
                        content=content
                    )
                    self.message_list.append(message)
                    
                
                else:  # REACTIVE                
                    speaker_agent.receive_message(self.message_list)
                    
                    response_content = speaker_agent.get_response()
                   
                    message = speaker_agent.send_message(
                        to=[agent.name for agent in listener_agents],
                        content=response_content
                    )
                    self.message_list.append(message)