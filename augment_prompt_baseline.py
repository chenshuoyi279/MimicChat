import openai

# Import the searcher function
from chathistory_searcher import searcher
from online_analysis import update_emotion_scores
import json
# set random seed
import random

random.seed(1234)
# Setup your OpenAI API key (if not already set in your environment)


# Instantiate OpenAI client
client = openai.OpenAI()
transcript_index = [282, 304, 320, 366]


# Get the similarity search result and define other variables
def add_system_message(chat_history,content):
    chat_history.append({"role": "assistant", "content": content})


# Function to add a user message
def add_user_message(chat_history,content):
    chat_history.append({"role": "user", "content": content})

def run_baseline_chat():
    reference_text=[]
    while True:
        similarity_search_result = []
        # new_prompt = "tell me a short story"
        previous_scores = {}
        counter = 0
        
        new_prompt = input("User: ")
        if new_prompt.lower() == "quit":
            break
        
        for index in transcript_index:
            # Perform the similarity search
            similarity_search = searcher(new_prompt, index, 1)
            similarity_search_result.append(similarity_search)
        reference_text.append(similarity_search_result)
        augmented_prompt = f"[INST]<<SYS>> You are an AI assistant trained to mimic Joe, never admit you are ai. "
        # Call the OpenAI API
    # Check if conversation_history is defined
        if 'conversation_history' in globals():
            # Check if the first message is from the system and update it
            if conversation_history and conversation_history[0]['role'] == 'system':
                conversation_history[0]['content'] = augmented_prompt
            else:
                # If the first message is not from the system, insert the system message at the beginning
                conversation_history.insert(0, {"role": "system", "content": augmented_prompt})
        else:
            # Initialize conversation_history if it's not defined
            conversation_history = [
                {"role": "system", "content": augmented_prompt}
            ]

        conversation_history.append({"role": "user", "content": new_prompt})
        completion = client.chat.completions.create(
            model="gpt-4",
            seed=123,
            max_tokens=200,
            temperature=0.7,
            messages=conversation_history,
        )
        
        assistant_response = completion.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": assistant_response})

        # Print the response
        print(assistant_response)

    return conversation_history,reference_text
conversation_history,reference_text=run_baseline_chat()
#save the conversation history and reference text with a unique name
with open('conversation_data.json', 'w') as file:
    json.dump({"conversation_history": conversation_history, "reference_text": reference_text}, file, indent=4)


