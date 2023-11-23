import openai

# Import the searcher function
from chathistory_searcher import searcher
#set random seed
import random
random.seed(1234)
# Setup your OpenAI API key (if not already set in your environment)


# Instantiate OpenAI client
client = openai.OpenAI()
transcript_index=[282,304,320,366]
# Get the similarity search result and define other variables
def add_system_message(content):
    chat_history.append({"role": "system", "content": content})

# Function to add a user message
def add_user_message(content):
    chat_history.append({"role": "user", "content": content})

similarity_search_result=[]
new_prompt = "tell me a short story"
for index in transcript_index:
    similarity_search = searcher(new_prompt,index,1)
    similarity_search_result.append(similarity_search)
    # print(similarity_search)

personality_traits = ""
language_register = ""
dominant_emotions = ""
conversation_context = ""
chat_history = []
add_user_message(new_prompt)
# Format the augmented prompt
augmented_prompt = f"[INST]<<SYS>> You are an AI assistant trained to mimic Joe based on chat history between Joe and Frank. Use the following pieces of retrieved context and analysis to generate a short conversation-like response that matches the speaker's tone and memory. Do not make up facts. Keep the response in line with the speaker's personality, language register, and emotional tone.<</SYS>>\n\nPrevious Chat History:\n- {similarity_search_result}\n\nAnalysis:\n- Personality Traits: {personality_traits}\n- Language Register: {language_register}\n- Dominant Emotions: {dominant_emotions}\n- Context: {conversation_context}\n\nThought: Consider the speaker's personality traits, language register, dominant emotions, and the current conversation's context to generate a response that matches the speaker's tone."
# augmented_prompt=""
# Call the OpenAI API
completion = client.chat.completions.create(
    model="gpt-4",
    seed=123,
    max_tokens=200,
    temperature=0.7,
    messages=[    {
      "role": "system",
      "content": augmented_prompt
    },
        {"role": "user", "content": new_prompt}
    ]
)
response = completion.choices[0].message.content
add_system_message(response)

# Print the response
print(response)
