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


# # ============================FAKE DATA============================
# @router.get("/predict")
# def get_load(message: str = Query(...)):
#     try:
# 
#         return fake_map[message]
#     except Exception as e:
#         # Log stack trace
#         logger.exception(e)
#         raise HTTPException(status_code=500, detail=str(e)) from e
# 
# 
# fake_map = {
#     "刘卫国是哪个公司的高管？":
#         {
#             "response": "刘卫国是 600148.SH 和 000030.SZ 公司的高管。",
#             "thought": "[1m> Entering new AgentExecutor chain...[0m\n[32;1m[1;3m{\n    \"action\": \"Cypher search\",\n    \"action_input\": \"刘卫国是哪个公司的高管？\"\n}[0m\n\n[1m> Entering new LLMCypherGraphChain chain...[0m\n[32;1m[1;3mGenerated Cypher statement:[0m\n[36;1m[1;3mMATCH p0=(n0:股票)<-[r0:任职于]-(n1:高管) \nWHERE n1.value='刘卫国'\nRETURN DISTINCT n0 AS n4 LIMIT 10;[0m\n\n[1m> Finished chain.[0m\n\nObservation: [36;1m[1;3m[<Node id=55793 labels={'股票'} properties={'value': '600148.SH'}>, <Node id=50071 labels={'股票'} properties={'value': '000030.SZ'}>][0m\nThought:[32;1m[1;3m{\n    \"action\": \"Final Answer\",\n    \"action_input\": \"刘卫国是 600148.SH 和 000030.SZ 公司的高管。\"\n}[0m\n\n[1m> Finished chain.[0m"
#         },
#     "海南有哪些上市公司？":
#         {
#             "response": "海南的上市公司有：600259.SH、000955.SZ、000505.SZ、300189.SZ、600238.SH、600221.SH、000886.SZ、000566.SZ、600759.SH、600515.SH。",
#             "thought": "[1m> Entering new AgentExecutor chain...[0m\n[32;1m[1;3m{\n    \"action\": \"Cypher search\",\n    \"action_input\": \"海南上市公司\"\n}[0m\n\n[1m> Entering new LLMCypherGraphChain chain...[0m\n[32;1m[1;3mGenerated Cypher statement:[0m\n[36;1m[1;3mMATCH p0=(n0:股票)-[r0:地域]->(n1:地域) WHERE n1.value='海南' \nRETURN DISTINCT n0 AS n4 LIMIT 10;[0m\n\n[1m> Finished chain.[0m\n\nObservation: [36;1m[1;3m[<Node id=55981 labels={'股票'} properties={'value': '600259.SH'}>, <Node id=50839 labels={'股票'} properties={'value': '000955.SZ'}>, <Node id=50225 labels={'股票'} properties={'value': '000505.SZ'}>, <Node id=53403 labels={'股票'} properties={'value': '300189.SZ'}>, <Node id=55951 labels={'股票'} properties={'value': '600238.SH'}>, <Node id=55919 labels={'股票'} properties={'value': '600221.SH'}>, <Node id=50739 labels={'股票'} properties={'value': '000886.SZ'}>, <Node id=50325 labels={'股票'} properties={'value': '000566.SZ'}>, <Node id=56779 labels={'股票'} properties={'value': '600759.SH'}>, <Node id=56361 labels={'股票'} properties={'value': '600515.SH'}>][0m\nThought:[32;1m[1;3m{\n    \"action\": \"Final Answer\",\n    \"action_input\": \"海南的上市公司有：600259.SH、000955.SZ、000505.SZ、300189.SZ、600238.SH、600221.SH、000886.SZ、000566.SZ、600759.SH、600515.SH。\"\n}[0m\n\n[1m> Finished chain.[0m"
#         },
#     "在北京由硕士学历的女性高管任职的上市公司，都属于哪些行业？":
#         {
#             "response": "在北京由硕士学历的女性高管任职的上市公司，属于家用电器、广告包装、化纤、建筑工程、家居用品、文教休闲、供气供热、运输设备、超市连锁、化工原料、汽车配件、中成药、通信设备、煤炭开采、装修装饰、百货、啤酒、生物制药、综合类、软件服务、区域地产、园区开发、全国地产、电信运营、IT设备、电气设备、小金属、酒店餐饮、普钢、饲料、汽车整车、铅锌、专用机械、新型电力、化学制药、航空、医疗保健、半导体、银行、矿物制品、服饰、证券、渔业、元器件、船舶、仓储物流、种植业、农业综合、水力发电、多元金融行业。",
#             "thought": "[1m> Entering new AgentExecutor chain...[0m\n[32;1m[1;3m{\n    \"action\": \"Cypher search\",\n    \"action_input\": \"在北京由硕士学历的女性高管任职的上市公司，都属于哪些行业？\"\n}[0m\n\n[1m> Entering new LLMCypherGraphChain chain...[0m\n[32;1m[1;3mGenerated Cypher statement:[0m\n[36;1m[1;3mMATCH \n  p1=(n1:`地域`)<-[:`地域`]-(n2:`股票`)<-[:`任职于`]-(n3:`高管`)-[:`性别`]->(n4:`性别`),\n  p2=(n3)-[:`学历`]->(n5:学历),\n  p3=(n2)-[:`所属行业`]->(n6:行业)\nWHERE n1.value='北京' AND n5.value='硕士' AND n4.value='F'\nRETURN DISTINCT n6.value AS hy;[0m\n\n[1m> Finished chain.[0m\n\nObservation: [36;1m[1;3m['家用电器', '广告包装', '化纤', '建筑工程', '家居用品', '文教休闲', '供气供热', '运输设备', '超市连锁', '化工原料', '汽车配件', '中成药', '通信设备', '煤炭开采', '装修装饰', '百货', '啤酒', '生物制药', '综合类', '软件服务', '区域地产', '园区开发', '全国地产', '电信运营', 'IT设备', '电气设备', '小金属', '酒店餐饮', '普钢', '饲料', '汽车整车', '铅锌', '专用机械', '新型电力', '化学制药', '航空', '医疗保健', '半导体', '银行', '矿物制品', '服饰', '证券', '渔业', '元器件', '船舶', '仓储物流', '种植业', '农业综合', '水力发电', '多元金融'][0m\nThought:[32;1m[1;3m{\n    \"action\": \"Final Answer\",\n    \"action_input\": \"在北京由硕士学历的女性高管任职的上市公司，属于家用电器、广告包装、化纤、建筑工程、家居用品、文教休闲、供气供热、运输设备、超市连锁、化工原料、汽车配件、中成药、通信设备、煤炭开采、装修装饰、百货、啤酒、生物制药、综合类、软件服务、区域地产、园区开发、全国地产、电信运营、IT设备、电气设备、小金属、酒店餐饮、普钢、饲料、汽车整车、铅锌、专用机械、新型电力、化学制药、航空、医疗保健、半导体、银行、矿物制品、服饰、证券、渔业、元器件、船舶、仓储物流、种植业、农业综合、水力发电、多元金融行业。\"\n}[0m\n\n[1m> Finished chain.[0m"
#         },
#     "李建国在哪些公司任职？":
#         {
#             "response": "根据查询结果，李建国曾在002065.SZ、002929.SZ、600123.SH、601825.SH、300369.SZ等公司任职。",
#             "thought": "[1m> Entering new AgentExecutor chain...[0m\n[32;1m[1;3m{\n    \"action\": \"Cypher search\",\n    \"action_input\": \"李建国 在哪些公司任职？\"\n}[0m\n\n[1m> Entering new LLMCypherGraphChain chain...[0m\n[32;1m[1;3mGenerated Cypher statement:[0m\n[36;1m[1;3mMATCH p0=(n0:股票)<-[r0:任职于]-(n1:高管) \nWHERE n1.value='李建国'\nRETURN DISTINCT n0 AS n4;[0m\n\n[1m> Finished chain.[0m\n\nObservation: [36;1m[1;3m[<Node id=51185 labels={'股票'} properties={'value': '002065.SZ'}>, <Node id=52833 labels={'股票'} properties={'value': '002929.SZ'}>, <Node id=59319 labels={'股票'} properties={'value': '688222.SH'}>, <Node id=59591 labels={'股票'} properties={'value': '688388.SH'}>, <Node id=55761 labels={'股票'} properties={'value': '600123.SH'}>, <Node id=58307 labels={'股票'} properties={'value': '603639.SH'}>, <Node id=57475 labels={'股票'} properties={'value': '601825.SH'}>, <Node id=53749 labels={'股票'} properties={'value': '300369.SZ'}>][0m\nThought:[32;1m[1;3m{\n    \"action\": \"Final Answer\",\n    \"action_input\": \"根据查询结果，李建国曾在002065.SZ、002929.SZ、600123.SH、601825.SH、300369.SZ等公司任职。\"\n}[0m\n\n[1m> Finished chain.[0m"
#         },
#     "2023年三月六日上市的股票有哪些？":
#         {
#             "response": "301322.SZ",
#             "thought": "[1m> Entering new AgentExecutor chain...[0m\n[32;1m[1;3m{\n    \"action\": \"Cypher search\",\n    \"action_input\": \"2023年三月六日上市的股票有哪些？\"\n}[0m\n\n[1m> Entering new LLMCypherGraphChain chain...[0m\n[32;1m[1;3mGenerated Cypher statement:[0m\n[36;1m[1;3mMATCH p0=(n0:股票)-[r0:上市日期]->(n1:上市日期) \nWHERE (n1.value>=20230306 AND n1.value<=20230306) \nRETURN DISTINCT n0 AS n4;[0m\n\n[1m> Finished chain.[0m\n\nObservation: [36;1m[1;3m[<Node id=55467 labels={'股票'} properties={'value': '301322.SZ'}>][0m\nThought:[32;1m[1;3m{\n    \"action\": \"Final Answer\",\n    \"action_input\": \"301322.SZ\"\n}[0m\n\n[1m> Finished chain.[0m"
#         },
#     "建筑工程行业有多少家上市公司？":
#         {
#             "response": "建筑工程行业共有上市公司115家。",
#             "thought": "[1m> Entering new AgentExecutor chain...[0m\n[32;1m[1;3m{\n    \"action\": \"Cypher search\",\n    \"action_input\": \"建筑工程行业上市公司数量\"\n}[0m\n\n[1m> Entering new LLMCypherGraphChain chain...[0m\n[32;1m[1;3mGenerated Cypher statement:[0m\n[36;1m[1;3mMATCH p0=(n0:股票)-[r0:所属行业]->(n1:行业) \nWHERE n1.value='建筑工程'\nRETURN COUNT(DISTINCT n0) AS n4;[0m\n\n[1m> Finished chain.[0m\n\nObservation: [36;1m[1;3m[115][0m\nThought:[32;1m[1;3m{\n    \"action\": \"Final Answer\",\n    \"action_input\": \"建筑工程行业共有上市公司115家。\"\n}[0m\n\n[1m> Finished chain.[0m"
#         },
#     "水泥行业博士学历的男性高管有多少位？":
#         {
#             "response": "水泥行业有46名具有博士学位的男性高管。",
#             "thought": "[1m> Entering new AgentExecutor chain...[0m\n[32;1m[1;3m{\n    \"action\": \"Cypher search\",\n    \"action_input\": \"水泥行业博士学历的男性高管有多少位？\"\n}[0m\n\n[1m> Entering new LLMCypherGraphChain chain...[0m\n[32;1m[1;3mGenerated Cypher statement:[0m\n[36;1m[1;3mMATCH \n  p0=(n1:行业)<-[r0:所属行业]-(n0:股票)<-[r1:任职于]-(n2:高管)-[r2:性别]->(n3:性别)-[r4:别名]->(n5:性别_别名),\n  p1=(n2)-[r3:学历]->(n4:学历) \nWHERE n1.value='水泥' AND n5.value='男性' AND n4.value='博士'\nRETURN COUNT(DISTINCT n2) AS n3;[0m\n\n[1m> Finished chain.[0m\n\nObservation: [36;1m[1;3m[46][0m\nThought:[32;1m[1;3m{\n    \"action\": \"Final Answer\",\n    \"action_input\": \"水泥行业有46名具有博士学位的男性高管。\"\n}[0m\n\n[1m> Finished chain.[0m"
#         }
# }

