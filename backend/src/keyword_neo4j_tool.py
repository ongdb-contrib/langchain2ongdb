from __future__ import annotations
from database import Neo4jDatabase
from langchain.chains.base import Chain

from typing import Any, Dict, List

from pydantic import Field
from logger import logger


fulltext_search = """
CALL db.index.fulltext.queryNodes("movie", $query) 
YIELD node, score
WITH node, score LIMIT 5
CALL {
  WITH node
  MATCH (node)-[r:!RATED]->(target)
  RETURN coalesce(node.name, node.title) + " " + type(r) + " " + coalesce(target.name, target.title) AS result
  UNION
  WITH node
  MATCH (node)<-[r:!RATED]-(target)
  RETURN coalesce(target.name, target.title) + " " + type(r) + " " + coalesce(node.name, node.title) AS result
}
RETURN result LIMIT 100
"""


def generate_params(input_str):
    """
    Generate full text parameters using the Lucene syntax
    """
    movie_titles = [title.strip() for title in input_str.split(',')]
    # Enclose each movie title in double quotes
    movie_titles = ['"' + title + '"' for title in movie_titles]
    # Join the movie titles with ' OR ' in between
    transformed_str = ' OR '.join(movie_titles)
    # Return the transformed string
    return transformed_str


class LLMKeywordGraphChain(Chain):
    """Chain for keyword question-answering against a graph."""

    graph: Neo4jDatabase = Field(exclude=True)
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.
        :meta private:
        """
        _output_keys = [self.output_key]
        return _output_keys

    def _call(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Extract entities, look up info and answer question."""
        question = inputs[self.input_key]
        params = generate_params(question)
        self.callback_manager.on_text(
            "Keyword query parameters:", end="\n", verbose=self.verbose
        )
        self.callback_manager.on_text(
            params, color="green", end="\n", verbose=self.verbose
        )
        logger.debug(f"Keyword search params: {params}")
        context = self.graph.query(
            fulltext_search, {'query': params})
        logger.debug(f"Keyword search context: {context}")
        return {self.output_key: context}


if __name__ == '__main__':
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=0.0)
    database = Neo4jDatabase(host="bolt://54.92.229.14:7687",
                             user="neo4j", password="adaptions-nod-prompts")
    chain = LLMKeywordGraphChain(llm=llm, verbose=True, graph=database)

    output = chain.run(
        "What type of movie is Top Gun and Matrix?"
    )

    print(output)
