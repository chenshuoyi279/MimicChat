from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Annoy


def searcher():
    # Create an instance of OpenAIEmbeddings
    embeddings = OpenAIEmbeddings()

    # Load the vector store
    vector_store = Annoy.load_local("/Users/evansmacbookpro/Desktop/MimicChat/304.ann", embeddings)

    # Define your new query
    query = "how is your mom?"

    # Perform the similarity search
    results = vector_store.similarity_search(query)

    # Helper function to get context
    def get_context(index, margin=5):
        start_index = max(0, index - margin)
        end_index = min(len(vector_store.docstore._dict), index + margin + 1)
        return start_index, end_index

    # Store printed indices to avoid duplication
    printed_indices = set()

    # List to store the augmented prompt components
    augmented_prompt_components = []

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
            content = document.page_content

            # Create the augmented prompt component
            component = f"Time: {time}, Speaker: {speaker}\n{content}\n{'-' * 80}"
            augmented_prompt_components.append(component)

    # Combine the augmented prompt components into a single string
    augmented_prompt = "\n\n".join(augmented_prompt_components)

    # Print the augmented prompt
    # print(augmented_prompt)


    return augmented_prompt
