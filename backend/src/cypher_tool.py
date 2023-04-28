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


if __name__ == "__main__":
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(
        openai_api_key=getEnv('OPENAI_KEY'),
        temperature=0.3)
    database = Neo4jDatabase(host="bolt://localhost:7687",
                             user="ongdb", password="123456")

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    readonlymemory = ReadOnlySharedMemory(memory=memory)

    chain = LLMCypherGraphChain(llm=llm, verbose=True, graph=database, memory=readonlymemory)

    output = chain.run(
        # "海南有哪些上市公司？"
        "在北京由硕士学历的女性高管任职的上市公司，都属于哪些行业？"
    )

    print(output)
