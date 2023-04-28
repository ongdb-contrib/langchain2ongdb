from backend.src.env import getEnv
from database import Neo4jDatabase
from pydantic import BaseModel, Extra
from langchain.prompts.base import BasePromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.base import Chain
from langchain.memory import ReadOnlySharedMemory, ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import Dict, List, Any

from logger import logger
import Levenshtein

examples = """
# 山西都有哪些上市公司？
MATCH p0=(n0:股票)-[r0:地域]->(n1:地域) WHERE n1.value='山西' 
RETURN DISTINCT n0 AS n4 LIMIT 10;

# 建筑工程行业有多少家上市公司？
MATCH p0=(n0:股票)-[r0:所属行业]->(n1:行业) 
WHERE n1.value='建筑工程'
RETURN COUNT(DISTINCT n0) AS n4;

# 火力发电行业博士学历的男性高管有多少位？
MATCH 
  p0=(n1:行业)<-[r0:所属行业]-(n0:股票)<-[r1:任职于]-(n2:高管)-[r2:性别]->(n3:性别)-[r4:别名]->(n5:性别_别名),
  p1=(n2)-[r3:学历]->(n4:学历) 
WHERE n1.value='火力发电' AND n5.value='男性' AND n4.value='博士'
RETURN COUNT(DISTINCT n2) AS n3;

# 在山东由硕士学历的男性高管任职的上市公司，都属于哪些行业？
MATCH 
  p1=(n1:`地域`)<-[:`地域`]-(n2:`股票`)<-[:`任职于`]-(n3:`高管`)-[:`性别`]->(n4:`性别`),
  p2=(n3)-[:`学历`]->(n5:学历),
  p3=(n2)-[:`所属行业`]->(n6:行业)
WHERE n1.value='山东' AND n5.value='硕士' AND n4.value='M'
RETURN DISTINCT n6.value AS hy;

# 2023年三月六日上市的股票有哪些？
MATCH p0=(n0:股票)-[r0:上市日期]->(n1:上市日期) 
WHERE (n1.value>=20230306 AND n1.value<=20230306) 
RETURN DISTINCT n0 AS n4 LIMIT 10;

# 2022年以来上市的股票有哪些？
MATCH p0=(n0:股票)-[r0:上市日期]->(n1:上市日期) 
WHERE n1.value>=20220101
RETURN DISTINCT n0 AS n4 LIMIT 10;

# 刘卫国是哪个公司的高管？
MATCH p0=(n0:股票)<-[r0:任职于]-(n1:高管) 
  WHERE n1.value='刘卫国'
RETURN DISTINCT n0 AS n4 LIMIT 10;
"""

SYSTEM_TEMPLATE = """
您是一名助手，能够根据示例Cypher查询生成Cypher查询。
示例Cypher查询是：\n""" + examples + """\n
不要回复除Cypher查询以外的任何解释或任何其他信息。
您永远不要为你的不准确回复感到抱歉，并严格根据提供的Cypher示例生成Cypher语句。
不要提供任何无法从密码示例中推断出的Cypher语句。
"""

SYSTEM_CYPHER_PROMPT = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE)

HUMAN_TEMPLATE = "{question}"
HUMAN_PROMPT = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)


class LLMCypherGraphChain(Chain, BaseModel):
    """Chain that interprets a prompt and executes python code to do math.
    """

    llm: Any
    """LLM wrapper to use."""
    system_prompt: BasePromptTemplate = SYSTEM_CYPHER_PROMPT
    human_prompt: BasePromptTemplate = HUMAN_PROMPT
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:
    graph: Neo4jDatabase
    memory: ReadOnlySharedMemory

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.
        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        logger.debug(f"Cypher generator inputs: {inputs}")
        chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_prompt] + inputs['chat_history'] + [self.human_prompt]
        )
        cypher_executor = LLMChain(
            prompt=chat_prompt, llm=self.llm, callback_manager=self.callback_manager
        )
        cypher_statement = cypher_executor.predict(
            question=inputs[self.input_key], stop=["Output:"])
        self.callback_manager.on_text(
            "Generated Cypher statement:", color="green", end="\n", verbose=self.verbose
        )
        self.callback_manager.on_text(
            cypher_statement, color="blue", end="\n", verbose=self.verbose
        )
        # If Cypher statement was not generated due to lack of context
        if not "MATCH" in cypher_statement:
            return {'answer': 'Missing context to create a Cypher statement'}
        context = self.graph.query(cypher_statement)
        logger.debug(f"Cypher generator context: {context}")

        return {'answer': context}


# def examples(ask):
#     examples_str = ''
#     examples_list = []
#     for index, map in enumerate(example_list()):
#         qa = map['qa']
#         cypher = map['cypher']
#         dis = Levenshtein.distance(ask, qa)
#         examples_list.append({'qa': qa, 'cypher': cypher, 'dis': dis})
#
#     sorted_list = sorted(examples_list, key=lambda map: map['dis'])
#
#     num = 0
#     for map in sorted_list:
#         qa = map['qa']
#         cypher = map['cypher']
#         ex = f'''
#           # {qa}
#           {cypher}
#           '''
#         if len(examples_str + ex) + 300 <= 2048:
#             examples_str += ex
#         num += 1
#         if num > 2:
#             break
#
#     return examples_str
#
#
# def example_list():
#     ex_list = [{
#         'qa': '火力发电行业博士学历的男性高管有多少位？',
#         'cypher': '''
#             MATCH
#                 p0=(n1:行业)<-[r0:所属行业]-(n0:股票)<-[r1:任职于]-(n2:高管)-[r2:性别]->(n3:性别)-[r4:别名]->(n5:性别_别名),
#                 p1=(n2)-[r3:学历]->(n4:学历)
#             WHERE n1.value='火力发电' AND n5.value='男性' AND n4.value='博士'
#             RETURN COUNT(DISTINCT n2) AS n3;
#             '''
#     }, {
#         'qa': '山西都有哪些上市公司？',
#         'cypher': '''
#             MATCH p0=(n0:股票)-[r0:地域]->(n1:地域) WHERE n1.value='山西'
#             RETURN DISTINCT n0 AS n4 LIMIT 10;
#                '''
#     }, {
#         'qa': '富奥股份的高管都是什么学历？',
#         'cypher': '''
#                  MATCH p0=(n1:股票名称)<-[r0:股票名称]-(n0:股票)<-[r1:任职于]-(n2:高管)-[r2:学历]->(n3:学历)
#             WHERE n1.value='富奥股份'
#             RETURN DISTINCT n3 AS n2 LIMIT 10;
#                    '''
#     }, {
#         'qa': '中国宝安属于什么行业？',
#         'cypher': '''
#                 MATCH p0=(n1:股票名称)<-[r0:股票名称]-(n0:股票)-[r1:所属行业]->(n2:行业)
#             WHERE n1.value='中国宝安'
#             RETURN DISTINCT n2 AS n5 LIMIT 10;
#                        '''
#     }, {
#         'qa': '建筑工程行业有多少家上市公司？',
#         'cypher': '''
#                 MATCH p0=(n0:股票)-[r0:所属行业]->(n1:行业)
#             WHERE n1.value='建筑工程'
#             RETURN COUNT(DISTINCT n0) AS n4;
#                            '''
#     }, {
#         'qa': '刘卫国是哪个公司的高管？',
#         'cypher': '''
#              MATCH p0=(n0:股票)<-[r0:任职于]-(n1:高管)
#             WHERE n1.value='刘卫国'
#             RETURN DISTINCT n0 AS n4 LIMIT 10;
#                             '''
#     }, {
#         'qa': '美丽生态上市时间是什么时候？',
#         'cypher': '''
#                 MATCH p0=(n1:股票名称)<-[r0:股票名称]-(n0:股票)-[r1:上市日期]->(n2:上市日期)
#             WHERE n1.value='美丽生态'
#             RETURN DISTINCT n2 AS n1 LIMIT 10;
#                                 '''
#     }, {
#         'qa': '山西的上市公司有多少家？',
#         'cypher': '''
#                 MATCH p0=(n0:股票)-[r0:地域]->(n1:地域)
#             WHERE n1.value='山西'
#             RETURN COUNT(DISTINCT n0) AS n4;
#                                     '''
#     }, {
#         'qa': '博士学历的高管都有哪些？',
#         'cypher': '''
#                 MATCH p0=(n0:高管)-[r0:学历]->(n1:学历)
#             WHERE n1.value='博士'
#             RETURN DISTINCT n0 AS n3 LIMIT 10;
#                                         '''
#     }, {
#         'qa': '上市公司是博士学历的高管有多少个？',
#         'cypher': '''
#                MATCH p0=(n0:高管)-[r0:学历]->(n1:学历)
#             WHERE n1.value='博士'
#             RETURN COUNT(DISTINCT n0) AS n3;
#                                             '''
#     }, {
#         'qa': '刘卫国是什么学历？',
#         'cypher': '''
#                 MATCH p0=(n0:高管)-[r0:学历]->(n1:学历)
#             WHERE n0.value='刘卫国'
#             RETURN DISTINCT n1 AS n2 LIMIT 10;
#                                                 '''
#     }, {
#         'qa': '富奥股份的男性高管有多少个？',
#         'cypher': '''
#                MATCH p0=(n1:股票名称)<-[r0:股票名称]-(n0:股票)<-[r1:任职于]-(n2:高管)-[r2:性别]->(n3:性别)-[r3:别名]->(n4:性别_别名)
#             WHERE n1.value='富奥股份' AND n4.value='男性'
#             RETURN COUNT(DISTINCT n2) AS n3;
#                                                     '''
#     }, {
#         'qa': '同在火力发电行业的上市公司有哪些？',
#         'cypher': '''
#                 MATCH p0=(n0:股票)-[r0:所属行业]->(n1:行业)
#             WHERE n1.value='火力发电'
#             RETURN DISTINCT n0 AS n4 LIMIT 10;
#                                                         '''
#     }, {
#         'qa': '同在火力发电行业的上市公司有多少家？',
#         'cypher': '''
#                 MATCH p0=(n0:股票)-[r0:所属行业]->(n1:行业)
#             WHERE n1.value='火力发电'
#             RETURN COUNT(DISTINCT n0) AS n4;
#                                                             '''
#     }, {
#         'qa': '大悦城和荣盛发展是同一个行业嘛？',
#         'cypher': '''
#                 MATCH p0=(n1:股票名称)<-[r0:股票名称]-(n0:股票)-[r1:所属行业]->(n2:行业)
#             WHERE n1.value IN ['大悦城','荣盛发展']
#             RETURN DISTINCT n2 AS n5 LIMIT 10;
#                                                                 '''
#     }, {
#         'qa': '同在河北的上市公司有哪些？',
#         'cypher': '''
#                 MATCH p0=(n0:股票)-[r0:地域]->(n1:地域)
#             WHERE n1.value='河北'
#             RETURN DISTINCT n0 AS n4 LIMIT 10;
#                                                                     '''
#     }, {
#         'qa': '神州高铁是什么时候上市的？',
#         'cypher': '''
#                MATCH p0=(n1:股票名称)<-[r0:股票名称]-(n0:股票)-[r1:上市日期]->(n2:上市日期)
#             WHERE n1.value='神州高铁'
#             RETURN DISTINCT n2 AS n1 LIMIT 10;
#                                                                        '''
#     }, {
#         'qa': '火力发电行业男性高管有多少个？',
#         'cypher': '''
#                MATCH p0=(n1:行业)<-[r0:所属行业]-(n0:股票)<-[r1:任职于]-(n2:高管)-[r2:性别]->(n3:性别)-[r3:别名]->(n4:性别_别名)
#             WHERE n1.value='火力发电' AND n4.value='男性'
#             RETURN COUNT(DISTINCT n2) AS n3;
#                                                                           '''
#     }, {
#         'qa': '2023年三月六日上市的股票代码？',
#         'cypher': '''
#                MATCH p0=(n0:股票)-[r0:上市日期]->(n1:上市日期)
#             WHERE (n1.value>=20230306 AND n1.value<=20230306)
#             RETURN DISTINCT n0 AS n4 LIMIT 10;
#                                                                              '''
#     }, {
#         'qa': '2022年以来上市的股票有哪些？',
#         'cypher': '''
#             MATCH p0=(n0:股票)-[r0:上市日期]->(n1:上市日期)
#             WHERE n1.value>=20220101
#             RETURN DISTINCT n0 AS n4 LIMIT 10;
#                                                                                  '''
#     }, {
#         'qa': '2023年三月六日上市的股票有哪些？',
#         'cypher': '''
#                MATCH p0=(n0:股票)-[r0:上市日期]->(n1:上市日期)
#             WHERE (n1.value>=20230306 AND n1.value<=20230306)
#             RETURN DISTINCT n0 AS n4 LIMIT 10;
#                                                                                  '''
#     }, {
#         'qa': '2023年三月六日上市的股票有多少个？',
#         'cypher': '''
#                 MATCH p0=(n0:股票)-[r0:上市日期]->(n1:上市日期)
#             WHERE (n1.value>=20230306 AND n1.value<=20230306)
#             RETURN COUNT(DISTINCT n0) AS n4;
#                                                                                      '''
#     }, {
#         'qa': '胡永乐是什么性别？',
#         'cypher': '''
#                  MATCH p0=(n0:高管)-[r0:性别]->(n1:性别)
#             WHERE n0.value='胡永乐'
#             RETURN DISTINCT n1 AS n7 LIMIT 10;
#                                                                                          '''
#     }, {
#         'qa': '在山东由硕士学历的男性高管任职的上市公司，都属于哪些行业？',
#         'cypher': '''
#                MATCH
#     	        p1=(n1:`地域`)<-[:`地域`]-(n2:`股票`)<-[:`任职于`]-(n3:`高管`)-[:`性别`]->(n4:`性别`),
#                 p2=(n3)-[:`学历`]->(n5:学历),
#                 p3=(n2)-[:`所属行业`]->(n6:行业)
#             WHERE n1.value='山东' AND n5.value='硕士' AND n4.value='M'
#             RETURN DISTINCT n6.value AS hy;
#                                                                                             '''
#     }]
#     return ex_list
#
# if __name__ == "__main__":
#     from langchain.chat_models import ChatOpenAI
#
#     llm = ChatOpenAI(
#         openai_api_key=getEnv('OPENAI_KEY'),
#         temperature=0.3)
#     database = Neo4jDatabase(host="bolt://localhost:7687",
#                              user="ongdb", password="123456")
#
#     memory = ConversationBufferMemory(
#         memory_key="chat_history", return_messages=True)
#     readonlymemory = ReadOnlySharedMemory(memory=memory)
#
#     message = "在北京由硕士学历的女性高管任职的上市公司，都属于哪些行业？"
#     # 动态调整Prompt
#     SYSTEM_TEMPLATE = """
#             您是一名助手，能够根据示例Cypher查询生成Cypher查询。
#             示例Cypher查询是：\n""" + examples(message) + """\n
#             不要回复除Cypher查询以外的任何解释或任何其他信息。
#             您永远不要为你的不准确回复感到抱歉，并严格根据提供的Cypher示例生成Cypher语句。
#             不要提供任何无法从密码示例中推断出的Cypher语句。
#             """
#     SYSTEM_CYPHER_PROMPT = SystemMessagePromptTemplate.from_template(SYSTEM_TEMPLATE)
#
#     HUMAN_TEMPLATE = "{question}"
#     HUMAN_PROMPT = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)
#
#     cypher_tool = LLMCypherGraphChain(
#         llm=llm, graph=database, verbose=True, memory=readonlymemory,
#         system_prompt=SYSTEM_CYPHER_PROMPT, human_prompt=HUMAN_PROMPT)
#     chain = LLMCypherGraphChain(llm=llm, verbose=True, graph=database, memory=readonlymemory)
#
#     output = chain.run(message)
#
#     print(output)
