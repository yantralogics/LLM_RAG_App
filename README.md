# Retrival Augmented Generation using Open AI LLM

## Key Highlights

This is a basic application that allows users to upload PDF files and chat with them. When running locally, after users upload the files and process them, the vector database will be persistent in the docs directory that they can query later as well. The conversation holds chat history as part of the prmopt context for better rendering.


## Technical Spec

The application uses following frameworks to make this happen.

 - Open AI gpt 3.5 Turbo model accessed via Open AI API (*Make sure you have a valid open AI api in the .env file )
 - Langchain ChatOpenAI, PromptTemplate & ConversationalRetrieval Chain, ConversationBufferMemory
 - Chroma vector store
 - Streamlit for user interface 
  


## Future Work

 - Exploration of alternative frameworks ( Llamaindex, Shiny for python)
 - Knowledge Graph in the RAG 



 ### Contact:

yantralogics@outlook.com
