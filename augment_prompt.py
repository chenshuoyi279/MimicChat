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
conversation_history=[]

# Get the similarity search result and define other variables
def add_system_message(chat_history,content):
    chat_history.append({"role": "assistant", "content": content})


# Function to add a user message
def add_user_message(chat_history,content):
    chat_history.append({"role": "user", "content": content})

def run_mimic_chat():
    reference_text=[]
    global conversation_history
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

            # Update the emotion scores with the results of the similarity search
            score = update_emotion_scores(similarity_search, previous_scores)

            # Update the previous scores for the next iteration
            previous_scores = score

            # Increment the counter
            counter += 1

            # Print the updated scores
            print(score)

            # Append the similarity search result to the list
            similarity_search_result.append(similarity_search)
            # print("Similarity Search Result:", similarity_search)
        reference_text.append(similarity_search_result)
        if counter > 0:
            average_scores = {speaker: total_score / counter for speaker, total_score in previous_scores.items()}
            print("Average Scores:", average_scores)

        personality_traits = ""
        language_register = ""
        dominant_emotions = average_scores
        conversation_context = ""
        # Format the augmented prompt
        augmented_prompt = f"[INST]<<SYS>> You are an AI assistant trained to mimic Joe based on chat history between Joe and Frank. Use the following pieces of retrieved context and analysis to generate a short conversation-like response that matches the speaker's tone and memory. Do not make up facts. Keep the response in line with the speaker's personality, language register, and emotional tone.<</SYS>>\n\nPrevious Chat History:\n- {similarity_search_result}\n\nAnalysis:\n- Personality Traits: {personality_traits}\n- Language Register: {language_register}\n- Dominant Emotions(higher score means more positive, lower score means more negative): {dominant_emotions}\n- Context: {conversation_context}\n\nThought: Consider the speaker's personality traits, language register, dominant emotions, and the current conversation's context to generate a response that matches the speaker's tone."
        # augmented_prompt=""
        # Call the OpenAI API
    # Check if conversation_history is defined
        if 'conversation_history' in globals() and conversation_history:
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

conversation_history,reference_text=run_mimic_chat()
#save the conversation history and reference text with a unique name
with open('conversation_data.json', 'w') as file:
    json.dump({"conversation_history": conversation_history, "reference_text": reference_text}, file, indent=4)


