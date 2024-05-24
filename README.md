# HDTwin: A Human Digital Twin using LLMs for Cognitive Diagnosis Prediction

## About the Repository
This repository stores accompanying data and code for the techniques presented in the research paper: G. Sprint, M. Schmitter-Edgecombe, and D. Cook. "HDTwin: Building a Human Digital Twin using Large Language Models for Cognitive Diagnosis Prediction" (currently under review).  

Files related to participant data and extracted decision tree (dt) rules in `data`:
* `train_synthetic.csv`: **synthetic training set** AKA reference group numeric and diagnosis data.
  * Note that this data has been synthetically generated from the real data to protect participant privacy.
* `test_synthetic.csv`: **synthetic test set** numeric data only
  * Note that this data has been synthetically generated from the real data to protect participant privacy.
  * Note that journal and interview transcript text data have been removed to protect participant privacy.
* `dt_rules.csv`: rules extracted from decision trees constructed using numeric markers.
  * Note that these rules were constructed using the original data, not the synthetic data and this discrepancy will affect classification performance.

The code is written in Python 3.11 and uses the following non-standard packages:
* [`numpy`](https://numpy.org/)
* [`pandas`](https://pandas.pydata.org/)
* [`scikit-learn`](https://scikit-learn.org/)
* [`faiss`](https://pypi.org/project/faiss-cpu/)
* [`openai`](https://github.com/openai/openai-python)
* [`langchain`](https://www.langchain.com/)
* [`streamlit`](https://streamlit.io/)

## Setup
1. Clone the repo
  ```sh
  git clone https://github.com/WSU-CASAS/HDTwin
  ```
2. Create a virtual environment
```sh
conda create --name hdtwin python=3.11 
conda activate hdtwin
```
3. Install Python packages
  ```sh
  pip install -r requirements.txt
  ```
4. Create an [OpenAI API key](https://openai.com/index/openai-api/)
5. Paste your API key in a `keys.json` file
```json
{
  "OPENAI_API_KEY": "YOUR API KEY HERE"
}
```
6. Follow usage instructions below


## Usage
1. **First setup vector stores**
1. Then either:  
    * Run Chatbot Agent in console or in streamlit app  
    * Run diagnosis classification over test set  
        * Note: classification accuracy will be poor due to synthetic test data (see note above)

### Setup Vector Stores
**Code files used to setup the vector stores** for the chatbot agent's retrieval augmented generation tools:
* `participant_retriever_tool.py`: loads `data/test.csv` and writes its contents into a FAISS database in `vector_stores/participant_vector_index`
* `knowledge_retriever_tool.py`: writes the knowledge base rules in the `rules` directory to a FAISS database in `vector_stores/knowledge_vector_index`

Example run to set up the vector stores (must be run before the chatbot because its tools use the vector stores):  
```sh
python agent_tools/participant_retriever_tool.py && python agent_tools/knowledge_retriever_tool.py
```

### Chatbot Agent
**Code files for setting up and running the chatbot agent**:
* `hdtwin_agent.py`: the HDTwin Chatbot Agent which is an OpenAI tools agent built with LangChain.
* `agent_tools`: directory with custom agent tools to support HDTwin functionality.
  * `participant_retriever_tool.py`: reads information about participants previously written to a vector store.
  * `knowledge_retriever_tool.py`: reads rules from a knowledge base previously written to a vector store.
  * `diagnosis_tool.py`: classifies a participant as healthy or mild cognitive impairment. Uses the rules referred to in the paper as the best wrapper rule set.
  * `reference_group_tool.py`: wraps a Pandas agent for on-the-fly calculation of summary statistics from a reference group AKA training set of participants.
* `hdtwin_app.py`: streamlit web app running the HDTwin Chatbot Agent locally in a GUI.

Example to run the chatbot agent through a few example prompts for a participant in the test set named Sloan:  
```sh
python hdtwin_agent.py -p Sloan
```

Example to launch the local streamlit web server for interacting with the chatbot agent via a GUI:  
```sh
streamlit run hdtwin_app.py
```

### Diagnosis Classification
Code files for setting up and running the chatbot agent presented in the research paper:
* `diagnosis_classification.py`: direct prompting of OpenAI models to classify test participants as healthy or MCI and calculate classification performance.
  * **Note that classification performance is not expected to match the reported results in the paper due to synthetically generated test data.** See note above about datasets.

Example to run diagnosis classification with the experiment label "example" for one run (AKA iteration). Creates a `results` directory and writes results csv files with "example" in the names to this directory:  
```sh
python diagnosis_classification.py -e example -n 1
```

## Notes on the OpenAI API
Predictions obtained from GPT models available via the OpenAI API may be different over time due to several reasons:
1. New models are released and old models are deprecated
1. Determinism is not guaranteed with large language models (the latest info regarding this will be available in the [API reference](https://platform.openai.com/docs/api-reference/chat)). Setting reproducibility parameters like `temperature` does not guarantee the same response for a prompt (at least not at the time of writing this README/code).
    * More details in [this OpenAI Developer Forum post](https://community.openai.com/t/why-the-api-output-is-inconsistent-even-after-the-temperature-is-set-to-0/329541)

## License
Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact
Gina Sprint - [@gsprint23](https://github.com/gsprint23) - sprint@gonzaga.edu