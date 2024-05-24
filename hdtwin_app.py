"""hdtwin_app.py
Streamlit app to run the HDTwin Chatbot Agent with a GUI locally.

@author Gina Sprint
@date 5/22/24
"""
import os
import json
import streamlit as st

from hdtwin_agent import HDTwinAgent

# STREAMLIT SESSION STATE
state = st.session_state

# SETUP AGENT CHAIN IF NOT ALREADY
def setup_chatbot_agent():    
    if "agent" not in state:
        print("Initializing agent chain...")
        # LOAD API KEY
        with open("keys.json") as infile:
            key_dict = json.load(infile)
            os.environ["OPENAI_API_KEY"] = key_dict["OPENAI_API_KEY"]
        # SAVE AGENT TO SESSION STATE SO RETAIN MESSAGE HISTORY
        state.agent = HDTwinAgent()

# PARSE AND DISPLAY AGENT RESPONSE
def generate_response(input_text):
    output_dict = state.agent.invoke(input_text)
    response = output_dict["output"]
    st.info(response)
    with st.expander("See Chat History"):
        chat_history = output_dict["chat_history"]
        st.write(chat_history)
    with st.expander("See Intermediate Steps"):
        intermediate_steps = output_dict["intermediate_steps"]
        st.write(intermediate_steps)
        
# SETUP AGENT AND UI COMPONENTS THEN RUN APP LOOP
def run_app():
    setup_chatbot_agent()

    # SETUP STREAMLIT COMPONENTS
    st.title("HDTwin Chatbot Agent")
    with st.form("submit_input_form"):
        text = st.text_area("Enter Input Text", height=200, placeholder="""Ask a question about a participant's health. Examples:
            Participant data retrieval: \"What is Sloan's shape_score_sd?\"
            Reference group data analysis: \"What is the average shape_score_sd?\"
            Participant test person diagnosis: \"Do you think Sloan has mild cognitive impairment?\"""")
        if st.form_submit_button("Submit"):
            generate_response(text)

# example run: streamlit run hdtwin_app.py
run_app()