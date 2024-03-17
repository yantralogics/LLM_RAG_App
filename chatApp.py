import streamlit as st
from dotenv import load_dotenv 
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import faiss
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from functions import retrieve_from_db, create_local_db, conversationChain, conversationChainWmemory
from streamlit_option_menu import option_menu

import os
import regex as re

def list_dir_files():
    files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    #print(files)
    return files
#@st.cache_data
def file_uploader_section():
    uploaded_file = st.file_uploader("Upload a PDF document",accept_multiple_files=True)

    if uploaded_file:
        st.success("File uploaded successfully!")
        # Process the uploaded file as needed
        
        return uploaded_file

def selectbox_section():
    ## Collections in the docs directory 
    files = os.listdir('./docs')
    
    #files = [f for f in os.listdir() if f.endswith(".pdf")]
    selected_file = st.selectbox("Select from available VectorStores", files)

    if selected_file:
        #st.success(f"Selected file: {selected_file}")
        # Process the selected file as needed
        return selected_file

def reset_conversation():
    st.session_state.chat_history = []
    st.session_state.firstTime = True
    st.session_state.dataLoaded = False
    st.session_state.messages = [
            {"role": "assistant", "content": "Hello there, am ChatGPT clone"}
    ]
    st.success("Chat session reset, the chat history has been re-initialized, you may also select a new data source")
        

#vectorDb = retrieve_from_db("myCollection")

def main():
    load_dotenv()
    #st.header("Chat with multiple pdfs :books:")

    st.set_page_config(page_title= "Chat with multiple pdfs",
                    page_icon=":books:")
    
    # check for messages in session and create if not exists
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello there, am ChatGPT clone"}
        ]

    if "chat_history" not in st.session_state.keys():
        ## Initiate the chat history:
        st.session_state.chat_history = []
        

    if "vectorDB" not in st.session_state.keys():
        st.session_state.vectorDB = None

    if "setup" not in st.session_state.keys():
        st.session_state.setup = []
    
    if "firstTime" not in st.session_state.keys():
        st.session_state.firstTime = True
    
    if "dataLoaded" not in st.session_state.keys():
        st.session_state.dataLoaded = False

    
     
    def menu_callback(key):
        selection = st.session_state[key]
        st.write(f"Selection changed to {selection}")
        

    with st.sidebar:
        selected = option_menu(menu_title="Main Menu",
                             options=["Instructions","Upload","Select"],
                             icons=['info-square-fill','upload','list-ul'],
                             menu_icon='gear', default_index=0,
                             on_change=menu_callback, key='menu_5', orientation='horizontal')
        
        selected




        ## Add a toggle switch:
        #upload__toggle = st.toggle('Upload File')

        file_uploader_container = st.empty()


        ## If the file is to be uploaded then you get the Upload documents path:
        if 1:
            if  selected == "Upload" and st.session_state.dataLoaded == False:
                #if st.session_state.vectorDB is None:
                collectionName = st.text_input("Name for the collection")
                uploaded_file = file_uploader_section()
                st.subheader("Upload Documents")
                
                


            ## process button 
                if st.button("Process"):
                    with st.spinner("Processing"):
                        ## Whenever the toggle is changed - the vectorDB needs to be reset
                        
                        localCollection = create_local_db(uploaded_file,collectionName)
                        st.session_state.vectorDB = None
                        vectorstore = retrieve_from_db(localCollection)
                        st.session_state.vectorDB = vectorstore
                        st.session_state.dataLoaded = True
                        st.success("File Processed, are you ready to chat?")
                        
                        
                        
            if selected == "Select" and st.session_state.dataLoaded == False:        
            ## OR select from the dropdown:
                #if st.session_state.vectorDB is None:
                preloaded_files = selectbox_section()
                selectDataSource = st.button("Select Data Source")
                if selectDataSource:
                ## reset the vectorDB 
                    st.session_state.vectorDB = None
                    vectorstore = retrieve_from_db(preloaded_files)
                    #st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.vectorDB = vectorstore
                    st.session_state.dataLoaded = True
                    st.success("Vector store updated")
                    st.write(f"You selected {preloaded_files}")

            st.session_state.firstTime = False
            
        reset = st.button("Reset Chat")
        if reset:
            reset_conversation()
    if selected == "Instructions":
        st.write("Here are the instructions")


        

        
        
    if st.session_state.dataLoaded and selected != "Instructions":
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        user_question = st.chat_input("Ask a question about your documents:")
        if user_question is not None:
            st.session_state.messages.append({"role":"user","content":user_question})
            with st.chat_message("user"):
                st.write(user_question)
            
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                    #ai_response = conversationChain(question=user_question, vectorDb=vectorDb)
                print(st.session_state.chat_history)
                ai_response = conversationChainWmemory(st.session_state.vectorDB,user_question,st.session_state.chat_history)

                st.write(ai_response["answer"])
            new_ai_response = {"role":"assistant","content":ai_response["answer"]}
            st.session_state.chat_history.extend([(user_question,ai_response["answer"])])
            st.session_state.messages.append(new_ai_response)
        view_messages = st.expander("View the message contents in session state")
        with view_messages:
        #view_messages.json(st.session_state.messages)
            """
                Message History initialized with:
                ```python
                msgs = StreamlitChatMessageHistory(key="langchain_messages")
                ```

                Contents of `st.session_state.langchain_messages`:
            """
            view_messages.json(st.session_state.messages)

    
    
    
    ## Condition to add if the vector store is not available, chat input should not occur

    
        
        
            
   


    




if __name__ == '__main__':
    main()


    