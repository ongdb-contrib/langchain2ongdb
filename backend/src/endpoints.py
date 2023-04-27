import logging

from agent import GraphAgent
from backend.src.env import getEnv
from database import Neo4jDatabase
from fastapi import APIRouter, HTTPException, Query
from run import get_result_and_thought_using_graph

neo4j_host = getEnv("NEO4J_URL")
neo4j_user = getEnv("NEO4J_USER")
neo4j_password = getEnv("NEO4J_PASS")
model_name = getEnv("MODEL_NAME")
# build router
router = APIRouter()
logger = logging.getLogger(__name__)
graph = Neo4jDatabase(
    host=neo4j_host, user=neo4j_user, password=neo4j_password)

# 不使用记忆组件功能，每次接口初始化
# agent_graph = GraphAgent.initialize(
#     graph=graph, model_name=model_name)


@router.get("/predict")
def get_load(message: str = Query(...)):
    try:
        agent_graph = GraphAgent.initialize(
            graph=graph, model_name=model_name)
        return get_result_and_thought_using_graph(agent_graph, message)
    except Exception as e:
        # Log stack trace
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e)) from e
