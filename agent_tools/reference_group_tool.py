"""reference_group_tool.py
Custom HDTwin tool that wraps a Pandas agent for on-the-fly calculation of summary
    statistics from a reference group AKA training set of participants.

@author Gina Sprint
@date 5/22/24
"""
import os
import json
import pandas as pd
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent 
from langchain_openai import ChatOpenAI
from langchain_core.tools import ToolException


PANDAS_AGENT_MODEL_NAME = "gpt-3.5-turbo-0125"

# REFERENCE GROUP TOOL INPUT
class ReferenceGroupToolInput(BaseModel):
    data_query: str = Field(description=f"""Query for reference group summary statistics.""")

# TOOL FUNCTIONS
def reference_group_tool_function(data_query: str) -> str:
    df = pd.read_csv(os.path.join("data", "train_synthetic.csv"))

    llm = ChatOpenAI(model=PANDAS_AGENT_MODEL_NAME,
                     temperature=0.0) 
    agent_executor = create_pandas_dataframe_agent(llm=llm,
                                          df=df,
                                          verbose=True,
                                          agent_type=AgentType.OPENAI_FUNCTIONS)
    text = agent_executor.invoke(data_query)

    if text is not None and text != "":
      return text
    else:
        raise ToolException(f"Reference group tool failed to answer this query.")

def _handle_forecast_error(error: ToolException) -> str:
    error_msg = "The following errors occurred during tool execution: "\
        + error.args[0]\
        + " Please try another tool."
    return error_msg

# REFERENCE GROUP TOOL
reference_group_struc_tool = StructuredTool.from_function(
    func=reference_group_tool_function,
    name="reference_data",
    description="""Useful for when you need data from reference group participants. When using this tool, you must first filter the data by \"mild cognitive impairment\" or \"healthy\".""",
    args_schema=ReferenceGroupToolInput,
    handle_tool_error=_handle_forecast_error,
)

if __name__ == "__main__":
    # example run: python agent_tools/reference_group_tool.py
    with open("keys.json") as infile:
        key_dict = json.load(infile)
        os.environ["OPENAI_API_KEY"] = key_dict["OPENAI_API_KEY"]

    print(reference_group_struc_tool.run({"data_query": "What is the average shape_score_sd for healthy participants?"}))