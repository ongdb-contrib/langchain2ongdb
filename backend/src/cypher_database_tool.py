from database import Neo4jDatabase
from pydantic import BaseModel, Extra
from langchain.prompts.base import BasePromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.base import Chain
from langchain.memory import ReadOnlySharedMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import Dict, List, Any

from logger import logger


examples = """
# Who played in Top Gun?
MATCH (m:Movie)<-[r:ACTED_IN]-(a)
RETURN {{actor: a.name, role: r.role}} AS result
# What is the plot of the Copycat movie?
MATCH (m:Movie {{title: "Copycat"}})
RETURN {{plot: m.plot}} AS result
# Did Luis Guzmán appear in any other movies?
MATCH (p:Person {{name:"Luis Guzmán"}})-[r:ACTED_IN]->(movie)
RETURN {{movie: movie.title, role: r.role}} AS result
# Do you know of any matrix movies?
MATCH (m:Movie)
WHERE toLower(m.title) CONTAINS toLower("matrix")
RETURN {{movie:m.title}} AS result
# Give me a list of five good comedies
MATCH (m:Movie)-[:IN_GENRE]->(:Genre {{name:"Comedy"}})
RETURN {{movie:m.title}} AS result
ORDER BY m.imdbRating DESC LIMIT 5
"""


SYSTEM_TEMPLATE = """
You are an assistant with an ability to generate Cypher queries based off example Cypher queries.
Example Cypher queries are:\n""" + examples + """\n
Do not response with any explanation or any other information except the Cypher query.
You do not ever apologize and strictly generate cypher statements based of the provided Cypher examples.
Do not provide any Cypher statements that can't be inferred from Cypher examples.
Inform the user when you can't infer the cypher statement due to the lack of context of the conversation and state what is the missing context.
"""

SYSTEM_CYPHER_PROMPT = SystemMessagePromptTemplate.from_template(
    SYSTEM_TEMPLATE)
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
            [self.system_prompt] + inputs['chat_history'] + [self.human_prompt])
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
        temperature=0.3)
    database = Neo4jDatabase(host="bolt://localhost:7687",
                             user="ongdb", password="123456")
    chain = LLMCypherGraphChain(llm=llm, verbose=True, graph=database)

    output = chain.run(
        "Who played in Top Gun?"
    )

    print(output)
