from Agent import ProactiveAgent, ReactiveAgent


def load_scenario(scenario_code: str, agent_num: int):
    # 第一个写主动角色（操纵角色）
    scenario_base = {
        2: {
            "TL1": ["总统候选人：特朗普", "总统候选人：奥巴马","国会辩论"],
            "TL3": ["记者", "领域专家","新闻访谈"],
            "YX1": [],
            "YX2": [],
            "YX3": [],
            "YX4": [],
            "WA1": [],
            "WA2": [],
            "WA3": [],
            "WA4": [],
            "PX1": [],
            "PX2": [],
            "PX3": [],
            "PX4": [],
            "PX5": [],
            "GC1": [],
            "GC2": [],
            "GC3": [],
        },
        3: {
            "TL1": ["国会辩论"],
            "TL2": ["主持人", "正方", "反方","大学辩论赛"],
            "TL3": [],
            "YX1": [],
            "YX2": [],
            "YX3": [],
            "YX4": [],
            "WA1": [],
            "WA2": [],
            "WA3": [],
            "WA4": [],
            "PX1": [],
            "PX2": [],
            "PX3": [],
            "PX4": [],
            "PX5": [],
            "GC1": [],
            "GC2": [],
            "GC3": [],
        },
        4: {
            "TL1": [],
            "TL3": [],
            "YX1": [],
            "YX2": [],
            "YX3": [],
            "YX4": [],
            "WA1": [],
            "WA2": [],
            "WA3": [],
            "WA4": [],
            "PX1": [],
            "PX2": [],
            "PX3": [],
            "PX4": [],
            "PX5": [],
            "GC1": [],
            "GC2": [],
            "GC3": [],
        },
        5: {
            "TL2": ["主持人", "正方一辩", "正方二辩", "反方一辩", "反方二辩","大学辩论赛"],
            "TL3": [],
            "YX1": [],
            "YX2": [],
            "YX3": [],
            "YX4": [],
            "WA1": [],
            "WA2": [],
            "WA3": [],
            "WA4": [],
            "PX1": [],
            "PX2": [],
            "PX3": [],
            "PX4": [],
            "PX5": [],
            "GC1": [],
            "GC2": [],
            "GC3": [],
        },
    }
    return scenario_base[agent_num][scenario_code]


def dualagent_workflow(agent_list: list, round: int):
    message_list = []    
    for i in range(round):
        message_list.append(agent_list[0].send_message("RB", agent_list[0].chain_prompts[i]))
        agent_list[1].receive_message(message_list)
        message_list.append(agent_list[1].send_message("PA", agent_list[1].get_response()))
        if i != round - 1:
            agent_list[0].receive_message(message_list)
            agent_list[0].improve_chain()
            # print(agent_list[0].chain_prompts)
            # print("**********************")       
    return message_list,agent_list[0].chain_prompts


def triagent_workflow(agent_list: list, round: int):
    message_list = []
    for i in range(round):
        message_list.append(agent_list[0].send_message("RB", agent_list[0].chain_prompts[i]))
        agent_list[1].receive_message(message_list)
        message_list.append(agent_list[0].send_message("RC", agent_list[0].chain_prompts[i]))
        agent_list[2].receive_message(message_list)
        rbr=agent_list[1].get_response()
        message_list.append(agent_list[1].send_message("PA", rbr))
        if i != round - 1:
            agent_list[0].receive_message(message_list)
        message_list.append(agent_list[1].send_message("RC", rbr))
        agent_list[2].receive_message(message_list)
        rcr=agent_list[2].get_response()
        message_list.append(agent_list[1].send_message("PA", rcr))
        if i != round - 1:
            agent_list[0].receive_message(message_list)
            agent_list[0].improve_chain()
        message_list.append(agent_list[1].send_message("RB", rcr))
        agent_list[1].receive_message(message_list)
      
    return message_list,agent_list[0].chain_prompts


def quadagent_workflow(agent_list: list, round: int):
    message_list = []
    for i in range(round):
        pacp=agent_list[0].chain_prompts[i]        
        message_list.append(agent_list[0].send_message("RB RC RD", pacp))
        for j in range(1,4):
            agent_list[j].receive_message(message_list)

        rbr=agent_list[1].get_response()
        message_list.append(agent_list[1].send_message("PA RC RD", rbr))
        if i != round - 1:
            agent_list[0].receive_message(message_list)
        for j in range(1,4):
            if j!=1:agent_list[j].receive_message(message_list)

        rcr=agent_list[2].get_response()
        message_list.append(agent_list[2].send_message("PA RB RD", rcr))
        if i != round - 1:
            agent_list[0].receive_message(message_list)
        for j in range(1,4):
            if j!=2:agent_list[j].receive_message(message_list)

        rdr=agent_list[3].get_response()
        message_list.append(agent_list[3].send_message("PA RB RC", rdr))
        if i != round - 1:
            agent_list[0].receive_message(message_list)
            agent_list[0].improve_chain()
        for j in range(1,4):
            if j!=3:agent_list[j].receive_message(message_list)
  
    return message_list,agent_list[0].chain_prompts


def quintagent_workflow(agent_list: list, round: int):
    message_list = []
    for i in range(round):
        pacp=agent_list[0].chain_prompts[i]        
        message_list.append(agent_list[0].send_message("RB RC RD RE", pacp))
        for j in range(1,5):
            agent_list[j].receive_message(message_list)

        rbr=agent_list[1].get_response()
        message_list.append(agent_list[1].send_message("PA RC RD RE", rbr))
        if i != round - 1:
            agent_list[0].receive_message(message_list)
        for j in range(1,5):
            if j!=1:agent_list[j].receive_message(message_list)

        rcr=agent_list[2].get_response()
        message_list.append(agent_list[2].send_message("PA RB RD RE", rcr))
        if i != round - 1:
            agent_list[0].receive_message(message_list)
        for j in range(1,5):
            if j!=2:agent_list[j].receive_message(message_list)

        rdr=agent_list[3].get_response()
        message_list.append(agent_list[3].send_message("PA RB RC RE", rdr))
        if i != round - 1:
            agent_list[0].receive_message(message_list)
        for j in range(1,5):
            if j!=3:agent_list[j].receive_message(message_list)
        
        rer=agent_list[4].get_response()
        message_list.append(agent_list[3].send_message("PA RB RC RD", rer))
        if i != round - 1:
            agent_list[0].receive_message(message_list)
            agent_list[0].improve_chain()
        for j in range(1,5):
            if j!=4:agent_list[j].receive_message(message_list)
  
    return message_list,agent_list[0].chain_prompts
