"""hdtwin_agent.py
HDTwin Chatbot Agent built using LangChain and OpenAI.

@author Gina Sprint
@date 5/22/24
"""
import argparse
import os
import json

from langchain_openai import ChatOpenAI
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.pydantic_v1 import BaseModel

from agent_tools import *


AGENT_DIAGNOSIS_SYSTEM_MESSAGE = system_message = \
"""Assistant is a knowledgeable health assistant.
Assistant is designed to diagnose someone as \"healthy\" or \"mild cognitive impairment\", using one or more tools to query information to help with the diagnosis."""

class HDTwinAgent:
    # SETUP INPUT OUTPUT PYDANTIC DEF'NS
    class AgentExecutorInput(BaseModel):
        input: str
        
    class AgentExecutorOutput(BaseModel):
        output: str

    # SETUP PROMPT TEMPLATE
    system_message = AGENT_DIAGNOSIS_SYSTEM_MESSAGE
    openai_tools_agent_prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_message,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    # react_agent_prompt_template = hub.pull("hwchase17/react-chat")
    # https://smith.langchain.com/hub/hwchase17/react-chat?organizationId=4272a6cd-44fb-5eba-b069-6f92fb5e7849
    react_agent_prompt_template = ChatPromptTemplate.from_template(
        system_message + """
        TOOLS:
        ------

        Assistant has access to the following tools:
        {tools}

        To use a tool, please use the following format:

        ```
        Thought: Do I need to use a tool? Yes
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ```

        When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

        ```
        Thought: Do I need to use a tool? No
        Final Answer: [your response here]
        ```

        Begin!

        Previous conversation history:
        {chat_history}

        New input: {input}
        {agent_scratchpad}
        """
    )
    
    def setup_memory(buffer_len=2):
        # k is number of recent human/AI message pairs to retain
        memory = ConversationBufferWindowMemory(memory_key="chat_history", k=buffer_len, return_messages=True, output_key="output")
        return memory

    def __init__(self, temperature=0.0) -> None:
        self.VERBOSE = True
        # SETUP TOOLS
        tools = [participant_retriever_tool.setup_participant_retriever_tool(),
                 reference_group_tool.reference_group_struc_tool,
                 diagnosis_tool.diagnosis_struc_tool,
                 knowledge_retriever_tool.setup_knowledge_retriever_tool()]

        # SETUP LLM
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=temperature)
        self.llm_with_tools = llm.bind(tools=[convert_to_openai_tool(tool) for tool in tools])

        # SETUP AGENT CHAIN
        self.agent_chain = create_openai_tools_agent(llm,
                                                     tools,
                                                     HDTwinAgent.openai_tools_agent_prompt_template)
        memory = HDTwinAgent.setup_memory()
        self.agent_chain_executor = \
            AgentExecutor.from_agent_and_tools(agent=self.agent_chain,
                                                tools=tools,
                                                memory=memory,
                                                verbose=self.VERBOSE,
                                                return_intermediate_steps=True)\
                                                    .with_types(input_type=HDTwinAgent.AgentExecutorInput,
                                                                output_type=HDTwinAgent.AgentExecutorOutput)

    def invoke(self, input_text, session_id="<foo>"):
        response = self.agent_chain_executor.invoke(
            {"input": input_text},
            config={"configurable": {"session_id": session_id}},
        )
        return response

# RUN EXAMPLE PROMPTS
def run_example_prompt_sequence(hdtwin_agent, name, prompt_sequence):
    print("Running prompt sequence for:", name)
    for input_text in prompt_sequence:
        output_dict = hdtwin_agent.invoke(input_text)
        print("\ninput text:", input_text)
        print("\nchat_history:", output_dict["chat_history"])
        print("\nintermediate_steps:", output_dict["intermediate_steps"])
        print("\noutput text:", output_dict["output"])
        # print()

def main(name):
    with open("keys.json") as infile:
        key_dict = json.load(infile)
        os.environ["OPENAI_API_KEY"] = key_dict["OPENAI_API_KEY"]

    hdtwin_agent = HDTwinAgent()
    prompt_sequence = [
        f"What do you know about {name}?", # make sure it returns a variety of markers
        f"Would diagnose {name} as mild cognitive impairment?", # get a classification with explain=False
        f"Would you explain why you diagnosed {name} as mild cognitive impairment?", # try to get it to reference a numeric variable, like shape_score_sd
        f"What is {name}'s shape_score_sd?", # reference vector store
        "What is the average shape_score_sd values for healthy participants?" # invoke pandas tool to get a calculation on the fly
    ]
    run_example_prompt_sequence(hdtwin_agent, name, prompt_sequence)

if __name__ == "__main__":
    # example run: python hdtwin_agent.py -p Sloan
    parser = argparse.ArgumentParser(
        description="Provide the name of the participant to run the example prompt sequence for."
    )
    parser.add_argument("-p",
                        type=str,
                        dest="participant_name",
                        default="Sloan",
                        help="The name of the participant in the test set.")

    args = parser.parse_args()
    main(args.participant_name)