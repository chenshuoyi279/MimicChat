# MimicChat

MimicChat is a RAG that uses the ChatGPT API to mimic the speaking of Joe or Frank from the podcast "The Basement Yard." 

## How to Use

### Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/chenshuoyi279/MimicChat.git
   cd MimicChat
   ```

2. **Set Up OpenAI API:**
   - Ensure you have an OpenAI API key. Set it up on your local machine by exporting the API key as an environment variable:
     ```bash
     export OPENAI_API_KEY='your-openai-api-key'
     ```

3. **Install Required Modules:**
   ```bash
   pip install langchain
   ```

### Running MimicChat

1. **Run MimicChat with Conversational Context and Emotional Analysis:**
   ```bash
   python3 augment_prompt.py
   ```

   - Each time you run `augment_prompt.py`, it updates the current conversation history store.

2. **Run Baseline MimicChat (No Conversational Context, No Emotional Analysis):**
   ```bash
   python3 augment_prompt_baseline.py
   ```

## Repository Structure

- `augment_prompt.py`: Script to run MimicChat with enhanced features.
- `augment_prompt_baseline.py`: Script to run the baseline version of MimicChat.
- `requirements.txt`: List of required Python modules for the project.
- `README.md`: Project documentation.
- `Transcripts`: The sourced transcripts of Basement Yard episodes, containing the transcribed conversations between Joe and Frank.
- `chathistory_searcher.py`: Searches the user input with the vector store of the annotated chat history and returns the relevant chat history.
- `conversation_data_baseline.json`: Chat history of the baseline model, updated each time `augment_prompt_baseline.py` is run.
- `conversation_data.json`: Chat history of the MimicChat model, containing the annotated chat history between Joe and Frank. User prompts and MimicChat responses are also appended.
- `evaluation.py`: Evaluates the model responses by generating cosine similarity, lexical diversity, and syntactic similarity.
- `online_analysis.py`: Evaluates the emotion using the VADER sentiment analyzer, used as part of the augmented prompt.


## Additional Information

- **Podcast Reference:** [The Basement Yard](https://www.youtube.com/@TheBasementYard)
```
