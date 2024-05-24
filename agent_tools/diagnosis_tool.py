"""diagnosis_tool.py
Custom HDTwin tool that classifies a participant as healthy or mild cognitive impairment.

@author Gina Sprint
@date 5/22/24
"""
import os
import re
import json
import numpy as np
import pandas as pd

from langchain.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain_core.tools import ToolException
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

DIAGNOSIS_TOOL_MODEL_NAME = "gpt-3.5-turbo-0125"
DIAGNOSIS_SYSTEM_MESSAGE = system_message = \
"""You are a knowledgeable health assistant designed to diagnose someone as \"healthy\" or \"mild cognitive impairment\" using diagnosis rules and relevant personal information."""
# LOAD PREVIOUSLY FOUND BEST WRAPPER RULES TO USE
rules_df = pd.read_csv(os.path.join("data", "dt_rules.csv"), index_col=0) 
BEST_WRAPPER_RULES = [rule for rule in rules_df.index if rules_df.loc[rule]["wrapper_best"]]
BEST_WRAPPER_RULE_COLS = []
for rule in BEST_WRAPPER_RULES: 
    BEST_WRAPPER_RULE_COLS.extend(eval(rules_df.loc[rule]["features"]))
BEST_WRAPPER_RULE_COLS = sorted(list(set(BEST_WRAPPER_RULE_COLS)))

# DIAGNOSIS TOOL INPUT
class DiagnosisToolInput(BaseModel):
    name: str = Field(description="Name of person to diagnosis")
    explain: bool = Field(description="Whether to explain the diagnosis (True) or not (False)")

# PROMPT TEMPLATING
def get_participant_info_str(name, df, cols_to_use):
    participant_ser = df.loc[name]
    info_str = ""
    for key in cols_to_use:
        val = participant_ser[key]
        info_str += f"{key}: {val}\n"
    info_str = info_str[:-1]
    return info_str

def build_rule_str(rules_to_use):
    rules_str = ""
    for i, rule in enumerate(rules_to_use):
        rules_str += f"Rule #{i + 1}: {rule}\n"
    rules_str = rules_str[:-1]
    return rules_str

def construct_diagnosis_prompt(name, participant_info_str, rules_str, explain=True):
    prompt = \
    f"""Diagnose {name} as \"healthy\" or \"mild cognitive impairment\" by applying the following rules to {name}'s information.

    Diagnosis Rules:
    {rules_str}

    {name}'s Information:
    {participant_info_str}

    Answer using the format:
    Diagnosis: ...
    """
    if explain:
        prompt += "Explanation: ..."
    # print("PROMPT:", prompt)
    return prompt

def construct_diagnosis_prompt_template():
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", DIAGNOSIS_SYSTEM_MESSAGE),
            ("human", "{input}"),
        ]
    )
    return prompt_template

# DIAGNOSIS CHAIN
def get_chain():
    prompt_template = construct_diagnosis_prompt_template()
    llm = ChatOpenAI(
                        model=DIAGNOSIS_TOOL_MODEL_NAME,
                        temperature=0.0,
                        model_kwargs={"seed": 0} # issues with 3.5 0125 model: https://community.openai.com/t/seed-param-and-reproducible-output-do-not-work/487245/25
                     )
    chain = prompt_template | llm | StrOutputParser()
    return chain

# DIAGNOSIS EXTRACTION
def extract_predicted_diagnosis(output_text):
    pred = np.nan
    pattern = r"Diagnosis:\s*(.*)"
    match = re.search(pattern, output_text)
    if match:
        diagnosis = match.group(1).lower()
        if diagnosis == "mild cognitive impairment" or "mild cognitive impairment" in diagnosis:
            pred = "mild cognitive impairment"
        elif diagnosis == "healthy" or  "healthy" in diagnosis:
            pred = "healthy"
    return pred

# RUN PROMPTING
def prompt_and_get_response(chain, name, df, rules_str, rules_cols_to_use, explain):
    participant_info_str = get_participant_info_str(name, df, rules_cols_to_use)
    prompt = construct_diagnosis_prompt(name, participant_info_str, rules_str, explain=explain)

    output_text = chain.invoke({
        "input": prompt
    })
    return prompt, output_text

def run_test_participants(explain):
    chain = get_chain()
    rules_str = build_rule_str(BEST_WRAPPER_RULES)

    df = pd.read_csv(os.path.join("data", "test_synthetic.csv"), index_col=0)
    actual_ser = df.pop("diagnosis")

    name_prompt_response_dict = {}
    pred_dict = {}
    for name in df.index:
        prompt, output_text = prompt_and_get_response(chain, name, df, rules_str, BEST_WRAPPER_RULE_COLS, explain)
        diagnosis = extract_predicted_diagnosis(output_text)        
        # print("RESPONSE:", output_text)
        # print("DIAGNOSIS:", diagnosis, "ACTUAL:", actual_df.loc[name]["diagnosis"])
        if pd.isnull(diagnosis):
            input("Diagnosis is null!! Press enter to flip coin...")
            diagnosis = "healthy" if np.random.randint(0, 2) == 0 else "mild cognitive impairment"
        pred_dict[name] = diagnosis
        name_prompt_response_dict[name] = {"prompt": prompt, "response": output_text, "pred": diagnosis, "actual": actual_ser.loc[name], "correct": diagnosis == actual_ser.loc[name]}
    name_prompt_response_df = pd.DataFrame(name_prompt_response_dict).T
    pred_ser = pd.Series(pred_dict)
    return name_prompt_response_df, pred_ser, actual_ser

# TOOL FUNCTIONS
def external_prompt_and_get_response(name, explain):
    chain = get_chain()
    rules_str = build_rule_str(BEST_WRAPPER_RULES)

    df = pd.read_csv(os.path.join("data", "test_synthetic.csv"), index_col=0)
    _, output_text = prompt_and_get_response(chain, name, df, rules_str, BEST_WRAPPER_RULE_COLS, explain)
    return output_text

def diagnosis_tool_function(name: str, explain: bool) -> str:
    response = external_prompt_and_get_response(name, explain)
    return response

def _handle_diagnosis_error(error: ToolException) -> str:
    error_msg = "The following errors occurred during tool execution: "\
        + error.args[0]\
        + " Please try another tool."
    return error_msg

# DIAGNOSIS TOOL
diagnosis_struc_tool = StructuredTool.from_function(
    func=diagnosis_tool_function,
    name="diagnosis",
    description="""Useful for when you need to diagnose someone as \"mild cognitive impairment\" or \"healthy\".
    name is the person you are diagnosing and explain is whether or not to to explain the diagnosis.
    If an explanation is not explicitly requested, it should default to False.
    """,
    args_schema=DiagnosisToolInput,
    handle_tool_error=_handle_diagnosis_error,
)

if __name__ == "__main__":
    # example run: python agent_tools/diagnosis_tool.py
    with open("keys.json") as infile:
        key_dict = json.load(infile)
        os.environ["OPENAI_API_KEY"] = key_dict["OPENAI_API_KEY"]

    # print(diagnosis_struc_tool.run({"name": "Sloan", "explain": False}))
    print(diagnosis_struc_tool.run({"name": "Sloan", "explain": True}))