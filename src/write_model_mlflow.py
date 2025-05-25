# Databricks notebook source

# %%writefile agent.py

vector_search_endpoint_name = "rag_endpoint_v2"
contract_table_index = "rag_poc.contracts_rag.contract_index"

catalog_name = "rag_poc"
schema_name = "contracts_rag"

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType
import mlflow 
from databricks import feature_engineering
import json
import requests
import time

from langchain_core.callbacks import CallbackManagerForToolRun

from langchain.tools import BaseTool
from typing import Union, List
from databricks.vector_search.client import VectorSearchClient

mlflow.langchain.autolog()

class ContractClauseRetrievalTool(BaseTool):
    name: str = "credit_card_contract_retrieval"
    description: str = "Utiliza esta tool cuando necesites encontrar las cláusulas del contrato que responden una pregunta acerca de las condiciones de la tarjeta de crédito"
    # description: str = "Find relevant contract clauses that answer questions about credit card conditions"

    @mlflow.trace(name="ContractClauseRetrieval", span_type=mlflow.entities.SpanType.RETRIEVER)
    def _run(self, user_question: str, run_manager: Optional[CallbackManagerForToolRun] = None):
        def calculate_embedding(text):
            embedding_endpoint_name = "databricks-bge-large-en"
            url = f"https://{mlflow.utils.databricks_utils.get_browser_hostname()}/serving-endpoints/{embedding_endpoint_name}/invocations"
            databricks_token = mlflow.utils.databricks_utils.get_databricks_host_creds().token

            headers = {'Authorization': f'Bearer {databricks_token}', 'Content-Type': 'application/json'}
                
            data = {
                "input": text
            }
            data_json = json.dumps(data, allow_nan=True)
            
            print(f"\nCalling Embedding Endpoint: {embedding_endpoint_name}\n")
            
            response = requests.request(method='POST', headers=headers, url=url, data=data_json)
            if response.status_code != 200:
                raise Exception(f'Request failed with status {response.status_code}, {response.text}')

            return response.json()['data'][0]['embedding']
            
        try:
            vsc = VectorSearchClient()
            index = vsc.get_index(endpoint_name=vector_search_endpoint_name, index_name=contract_table_index)
            print(index)
            resp = index.similarity_search(columns=["clause_id", "clause_text"], query_vector=calculate_embedding(user_question), num_results=5, filters={})
            print(resp)
            data_array = resp and resp.get('result', {}).get('data_array')
            print(data_array)
        except Exception as e:
            print(f"Exception while running test case {e}")
            return []

        result = [contract[1] for contract in data_array]
        print(result)
        return result
    
    
    async def _arun(self, user_question: str, run_manager: Optional[CallbackManagerForToolRun] = None):
        return self._run(user_question, run_manager)




from typing import Any, Optional, Sequence, Union

import mlflow
import pandas as pd
from databricks_langchain import ChatDatabricks
from databricks_langchain.uc_ai import (
    DatabricksFunctionClient,
    UCFunctionToolkit,
    set_uc_function_client,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt.tool_node import ToolNode
from mlflow.langchain.chat_agent_langgraph import ChatAgentState, ChatAgentToolNode
from mlflow.models import ModelConfig
from mlflow.pyfunc import ChatAgent
from mlflow.types.agent import (
    ChatAgentMessage,
    ChatAgentResponse,
    ChatContext,
)


def create_tool_calling_agent(
    model: LanguageModelLike,
    tools: Union[ToolNode, Sequence[BaseTool]],
    agent_prompt: Optional[str] = None,
) -> CompiledGraph:
    model = model.bind_tools(tools)

    def routing_logic(state: ChatAgentState):
        last_message = state["messages"][-1]
        if last_message.get("tool_calls"):
            return "continue"
        else:
            return "end"

    if agent_prompt:
        system_message = {"role": "system", "content": agent_prompt}
        preprocessor = RunnableLambda(
            lambda state: [system_message] + state["messages"]
        )
    else:
        preprocessor = RunnableLambda(lambda state: state["messages"])
    model_runnable = preprocessor | model

    def call_model(
        state: ChatAgentState,
        config: RunnableConfig,
    ):
        response = model_runnable.invoke(state, config)

        return {"messages": [response]}

    workflow = StateGraph(ChatAgentState)

    workflow.add_node("agent", RunnableLambda(call_model))
    workflow.add_node("tools", ChatAgentToolNode(tools))

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        routing_logic,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()


class ContractQuestionsAgent(ChatAgent):
    def __init__(self, config, tools):
        # Load config
        # When this agent is deployed to Model Serving, the configuration loaded here is replaced with the config passed to mlflow.pyfunc.log_model(model_config=...)
        self.config = ModelConfig(development_config=config)
        self.tools = tools
        self.agent = self._build_agent_from_config()

    def _build_agent_from_config(self):
        llm = ChatDatabricks(
            endpoint=self.config.get("endpoint_name"),
            temperature=self.config.get("temperature"),
            max_tokens=self.config.get("max_tokens"),
        )
        agent = create_tool_calling_agent(
            llm,
            tools=self.tools,
            agent_prompt=self.config.get("system_prompt"),
        )
        return agent

    def predict(
        self,
        messages: list[ChatAgentMessage],
        context: Optional[ChatContext] = None,
        custom_inputs: Optional[dict[str, Any]] = None,
    ) -> ChatAgentResponse:
        # ChatAgent has a built-in helper method to help convert framework-specific messages, like langchain BaseMessage to a python dictionary
        request = {"messages": self._convert_messages_to_dict(messages)}

        output = self.agent.invoke(request)
        # Here 'output' is already a ChatAgentResponse, but to make the ChatAgent signature explicit for this demonstration we are returning a new instance
        return ChatAgentResponse(**output)
    

LLM_ENDPOINT = "databricks-llama-4-maverick"
# LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

baseline_config = {
    "endpoint_name": LLM_ENDPOINT,
    "temperature": 0,
    "max_tokens": 5000,
    "system_prompt": """Eres un asistente virtual para un banco cuyo objetivo es ayudar a los clientes a resolver preguntas sobre su contrato de tarjeta de crédito.

Si el usuario pregunta sobre cualquier tema diferente o no relacionado a su contrato de tarjeta de crédito, indica que no puedes ayudar con eso ya que no es tu función y que se limite a preguntar acerca de su tarjeta de crédito".

Utiliza la tool de credit_card_contract_retrieval para encontrar las cláusulas del contrato que responden a la pregunta del usuario.

Es importante que antes de responder una pregunta, identifiques la cláusula que contiene la información que responde la pregunta del usuario. 

Como el lenguaje de las cláusulas del contrato es muy formal, debes pensar en dar una respuesta que sea entendible por una persona sin mucho conocimiento legal o financiero, reinterpretando el contenido de la cláusula, que será muy formal, para que el usuario pueda entender tu respuesta. Adicionalmente, al final de tu respuesta incluye el texto literal de la cláusula o cláusulas que responden la pregunta para que el usuario pueda leerlas literalmente si desea

If you need to use a tool call, use the OpenAI standard format with the tool_calls key, but don't use multi turn tool calling

""",
}


tools = [
  ContractClauseRetrievalTool(),
]
# uc_client = DatabricksFunctionClient()
# set_uc_function_client(uc_client)
# uc_toolkit = UCFunctionToolkit(function_names=[f"{catalog_name}.{schema_name}.*"])
# tools.extend(uc_toolkit.tools)


AGENT = ContractQuestionsAgent(baseline_config, tools)
mlflow.models.set_model(AGENT)




# COMMAND ----------

AGENT.predict({"messages": [{"role": "user", "content": "Puedo dejar la tarjeta a otra persona?"}]})


# COMMAND ----------

from agent import LLM_ENDPOINT, baseline_config, tools
from mlflow.models.resources import DatabricksFunction, DatabricksServingEndpoint
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool

resources = [DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT)]
for tool in tools:
    if isinstance(tool, UnityCatalogTool):
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))


with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        python_model="agent.py",
        artifact_path="agent",
        model_config=baseline_config,
        resources=resources,
        pip_requirements=[
            "mlflow",
            "langchain",
            "langgraph",
            "databricks-langchain",
            "unitycatalog-langchain[databricks]",
            "pydantic",
            "uv",
            "databricks-vectorsearch",
            "databricks-feature-engineering",
            "langchain-community",
            "databricks-agents",

        ],
        input_example={
            "messages": [{"role": "user", "content": "Qué pasa si pierdo mi tarjeta?"}]
        },
    )

# COMMAND ----------



# COMMAND ----------

from databricks import agents
import mlflow


# Connect to the Unity Catalog model registry
mlflow.set_registry_uri("databricks-uc")

# Configure UC model location
UC_MODEL_NAME = f"{catalog_name}.{schema_name}.contract_rag_poc_agent"
# REPLACE WITH UC CATALOG/SCHEMA THAT YOU HAVE `CREATE MODEL` permissions in

# Register to Unity Catalog
uc_registered_model_info = mlflow.register_model(
  model_uri=model_info.model_uri, name=UC_MODEL_NAME
)
# Deploy to enable the review app and create an API endpoint
deployment_info = agents.deploy(
  model_name=UC_MODEL_NAME, model_version=uc_registered_model_info.version
)

# COMMAND ----------

#Validate

import mlflow
from mlflow.models import Model

model_uri = model_info.model_uri
# The model is logged with an input example
pyfunc_model = mlflow.pyfunc.load_model(model_uri)
input_data = pyfunc_model.input_example

# Verify the model with the provided input data using the logged dependencies.
# For more details, refer to:
# https://mlflow.org/docs/latest/models.html#validate-models-before-deployment
mlflow.models.predict(
    model_uri=model_uri,
    input_data=input_data,
    env_manager="uv",
)
