from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Annoy


def searcher(query, file_name,margin=3):
    # Create an instance of OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()

    # Load the vector store
    vector_store = Annoy.load_local(f"transcripts/{file_name}.ann", embeddings)

    # Perform the similarity search
    results = vector_store.similarity_search(query)

    # Helper function to get context
    def get_context(index, margin=margin):
        start_index = max(0, index - margin)
        end_index = min(len(vector_store.docstore._dict), index + margin + 1)
        return start_index, end_index

    # Define a format for each result component
    def format_component(time, speaker, content):
        return f"[Time: {time}, Speaker: {speaker}] {content}"

    # List to store formatted components
    formatted_components = []

    # Store printed indices to avoid duplication
    printed_indices = set()

    # Process the search results with context
    for result in results:
        # Find the index for the result
        for index, doc_id in vector_store.index_to_docstore_id.items():
            if vector_store.docstore._dict[doc_id] == result:
                break

        # Get context indices
        start_index, end_index = get_context(index)

        # Check for redundancy
        if any(i in printed_indices for i in range(start_index, end_index)):
            continue

        # Update printed indices
        printed_indices.update(range(start_index, end_index))

        # Process the context
        for context_index in range(start_index, end_index):
            doc_id = vector_store.index_to_docstore_id[context_index]
            document = vector_store.docstore._dict[doc_id]
            speaker = document.metadata.get('speaker')
            time = document.metadata.get('time')
            content = document.page_content.strip()

            # Format and add the component to the list
            formatted_components.append(format_component(time, speaker, content))

    # Combine the formatted components into a single string
    formatted_output = "\n".join(formatted_components)

    return formatted_output

# Example usage
formatted_prompt = searcher("hows ur mom", 304,3)
# print(formatted_prompt)
