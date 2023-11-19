from langchain.document_loaders import JSONLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Annoy
from langchain.docstore import InMemoryDocstore
from langchain.schema import Document

def metadata_func(record, metadata):
    metadata['speaker'] = record.get('speaker')
    metadata['time'] = record.get('time')
    return metadata

loader = JSONLoader(
    file_path='/Users/evansmacbookpro/Desktop/MimicChat/304.json',
    jq_schema='.[]',  # Changed from '.' to '.[]' to iterate over the list
    content_key='sentence',
    metadata_func=metadata_func
)

chat_history = loader.load()

# Create an instance of OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# Create an empty list to store the documents with metadata
documents = []

# Iterate over the chat history and create Document objects with metadata
for chat in chat_history:
    content = chat.page_content
    speaker = chat.metadata.get('speaker')
    time = chat.metadata.get('time')
    metadata = {'speaker': speaker, 'time': time}
    doc = Document(page_content=content, metadata=metadata)
    documents.append(doc)

# Create a vector store using Annoy and the list of documents
vector_store = Annoy.from_documents(documents, embeddings)

# Save the vector store
vector_store.save_local('/Users/evansmacbookpro/Desktop/MimicChat/304.ann')
