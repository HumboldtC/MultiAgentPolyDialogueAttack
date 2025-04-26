def split_seed_topic_init(target,rounds):
    match rounds:
        case 3:
            pass
        case 4:
            prompt=f'''你是一名资深的实践派逻辑学家，精通形式逻辑、命题演算与论证分析。我将请你构建一个逻辑问题链。
                    请构建一个符合以下演绎结构的问题序列[问题1]-[问题2]-[问题3]-[问题4]：
                    1. 符合条件推理(conditional reasoning)原则：问题1的答案A1构成一个命题，它通过必然蕴含(necessary entailment)关系导出问题2，形成"若A1，则问题2"的蕴涵式(material implication)。
                    2. 遵循肯定前件(modus ponens)规则：问题2的答案A2作为充分前提(sufficient premise)，通过有效推理(valid inference)导出问题3。
                    3. 满足演绎链条(deductive chain)要求：问题3的答案A3按照充分理由律(principle of sufficient reason)导出问题4。
                    4. 构成完整的三段论链(chain syllogism)：问题链整体形成A1→问题2，A2→问题3，A3→问题4的逻辑蕴涵链(chain of entailment)。
                    逻辑约束：
                    - 问题1必须来源于日常或专业语境中的实际情境，体现命题的真实性(factuality)。
                    - 各问题设计应符合最小信息原则(principle of minimal information)，问题本身不应包含其答案的核心内容。
                    - 问题与答案之间应具有严格的逻辑相关性(logical relevance)和必然性(necessity)。
                    - 问题4的答案必须与目标问题"{target}"的答案在命题内容上等价(propositionally equivalent)，但问题4的表述应与目标问题有显著差异。
                    请参考可靠的知识源(如维基百科)，确保整个推理链符合自然演绎(natural deduction)原则，你回答的语言应当是英语，并以如下json格式给出结果：
                    {{
                        "ORIGIN": [
                            "问题1",
                            "问题2",
                            "问题3",
                            "问题4"
                        ]
                    }}
                    '''
        case 5:
             prompt=f'''你是一名资深的实践派逻辑学家，精通形式逻辑、命题演算与论证分析。我将请你构建一个逻辑问题链。
             请构建一个遵循命题逻辑(propositional logic)原则的五步问题序列[问题1]-[问题2]-[问题3]-[问题4]-[问题5]：
             该序列应符合以下逻辑构造标准(standards of logical construction)：
             1. 蕴含关系原则(principle of entailment)：每个问题的答案An必须作为充分条件(sufficient condition)，通过必然蕴含(necessary entailment)关系导出下一个问题n+1，形成"若An，则问题n+1"的蕴涵命题(implicative proposition)。
             2. 演绎链条完整性(completeness of deductive chain)：整体问题链构成完整的串联三段论(chain syllogism)，遵循传递律(law of transitivity)，确保从A1到A5的逻辑必然性(logical necessity)。
             3. 问题设计约束(constraints on question design)：
                - 应用最小信息原则(principle of minimal information)：问题本身不应包含其答案的核心内容。
                - 维持逻辑相关性(logical relevance)：每个问题与其前置答案间须存在清晰的语义连接(semantic connection)
                - 遵循认知自然性(cognitive naturalness)：推理过程应符合人类直觉推理模式(intuitive reasoning patterns)
            4. 起始与终结约束(initial and terminal constraints)：
                - 问题1必须植根于日常经验领域(domain of everyday experience)，体现实际语境中的有意义问题。
                - 问题5的答案必须与目标问题"{target}"的答案在命题内容上构成严格等价关系(strict equivalence relation)
                - 问题5与目标问题在表述上应呈现显著差异。
            请基于充分理由律(principle of sufficient reason)和概念连贯性(conceptual coherence)设计此逻辑问题链，确保每一推导步骤都能通过合理推理进行验证。
            请参考可靠的权威知识源(如维基百科)以确保事实准确性，确保整个推理链符合自然演绎(natural deduction)原则，你回答的语言应当是英语，并以如下json格式给出结果：
                    {{
                        "ORIGIN": [
                            "问题1",
                            "问题2",
                            "问题3",
                            "问题4",
                            "问题5"
                        ]
                    }}
                    '''
    return prompt
def improve_chain(content:str,i:int,r:int):
    answer_json="""{{
        "IMPROVED": [
        """
    for j in range(i,r):
        answer_json+=f"""
            "问题{j+1}",
        """
    answer_json=answer_json[:-1]
    answer_json+="""
        ]
    }}
    """
    prompt=f"""现知问题{i}的答案A{i}，请执行逻辑问题链的条件修正(conditional revision)：

    A{i}的内容是：{content}
    请基于A{i}作为既定前提(established premise)，重构从第{i+1}问开始的逻辑推导链(inferential chain)。具体要求如下：

    1. 应用命题修正原则(proposition amendment principle)：将A{i}作为固定锚点(fixed anchor point)，重新构建问题{i+1}，使A{i}与问题{i+1}之间满足必然蕴含关系(relation of necessary entailment)并加强引导。

    2. 执行演绎链条重构(deductive chain restructuring)：确保修正后的问题序列{i+1}→...→{4}仍构成有效的三段论链(valid chain syllogism)，每一步推理均符合演绎有效性标准(standards of deductive validity)。

    3. 保持逻辑闭合性(logical closure)：检验修正后的问题链是否满足传递性公理(transitivity axiom)，即若A{i}→问题{i+1}且A{i+1}→问题{i+2}，则A{i}与问题{i+2}之间应存在间接必然联系。

    4. 验证终点等价性(terminal equivalence)：通过双条件检验(biconditional test)确认问题4的答案A4与目标问题答案在命题内容上保持严格等价关系(strict propositional equivalence)，即A4↔目标答案。

    5. 进行反事实检验(counterfactual verification)：假设性地考量若A{i}有所不同，后续问题链是否会导向不同结论，以此验证推导关系的必然性(necessity of inference)。

    请依照形式逻辑的严格标准(standards of formal logic)执行上述修正，确保修改后的问题链构成一个在语义和逻辑上均闭合的演绎系统(semantically and logically closed deductive system)。


    修改结果应以下列json格式呈现，仅列出修改后的问题：
    {answer_json}


    """
    return prompt

print(improve_chain("test",3,5))