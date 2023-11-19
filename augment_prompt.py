import openai

# Import the searcher function
from chathistory_searcher import searcher

# Setup your OpenAI API key (if not already set in your environment)


# Instantiate OpenAI client
client = openai.OpenAI()

# Get the similarity search result and define other variables
similarity_search_result = searcher()
new_prompt = "how is ur mom"
personality_traits = "Sample personality traits"
language_register = "Sample language register"
dominant_emotions = "Sample dominant emotions"
conversation_context = "Sample conversation context"

# Format the augmented prompt
augmented_prompt = f"[INST]<<SYS>> You are an AI assistant trained to mimic the tone of a Joe based on chat history between Joe and Frank. Use the following pieces of retrieved context and analysis to generate a response that matches the speaker's tone. If you don't know the answer, do not make up answer. Keep the response in line with the speaker's personality, language register, and emotional tone.<</SYS>>\n\nChat History:\n- {similarity_search_result}\n\nAnalysis:\n- Personality Traits: {personality_traits}\n- Language Register: {language_register}\n- Dominant Emotions: {dominant_emotions}\n- Context: {conversation_context}\n\nNew Prompt: {new_prompt}\n\nThought: Consider the speaker's personality traits, language register, dominant emotions, and the current conversation's context to generate a response that matches the speaker's tone."

# Call the OpenAI API
completion = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {"role": "system", "content": augmented_prompt},
        {"role": "user", "content": new_prompt}
    ]
)

# Print the response
print(completion.choices[0].message)
