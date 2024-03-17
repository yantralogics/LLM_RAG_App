import os
import openai
import sys
#sys.path.append('../..')

import panel as pn  # GUI
pn.extension()

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']

from sqlite3 import Row
#from tkinter.tix import COLUMN
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch, Chroma
#from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.prompts import PromptTemplate, MessagesPlaceholder ,ChatPromptTemplate


def create_local_db(filenames,collectionName):
    ## takes in input pdf filenames
    ## loads them using the PyPDF loader
    ## splits them using the text splitter
    ## saves them in the Chroma db
    print("Calling create_local_db")
    text = ""
    if type(filenames) is str:
        pdf__reader = PdfReader(filenames)
        for page in pdf__reader.pages:
            # append text variable
            text += page.extract_text()
    else:
        print(len(filenames))
        for pdf in filenames:
        # initiate pdf reader object
            pdf__reader = PdfReader(pdf)

        # loop over the pdf pages
            for page in pdf__reader.pages:
            # append text variable
                text += page.extract_text()


    # loop
    
    
    ## Extracted text
    text_splitter= CharacterTextSplitter(
        separator= "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings()

    dirName = f'docs/{collectionName}'    

    vectorDB = Chroma.from_texts(
        texts = chunks,
        embedding = embeddings,
        persist_directory = dirName
    )
    print("Vector saved to files")

    return 1


def retrieve_from_db(collectionName):
    ## Collection name has the contents of embedded data(set)
    print("Calling retrieve from db")
    ## Get the directory
    persist_directory  = f"docs/{collectionName}"
    ## get the embedding information
    embedding = OpenAIEmbeddings()
    ## retrieve the db
    vectordb = Chroma(persist_directory = persist_directory,
                    embedding_function = embedding)
    print("Vector loaded from file")
    return vectordb


def conversationChain(question,vectorDb):
    ## This is the function that gets the response 
    
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
                {context}
                Question: {question}
                Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)
    ### We can pass in the prompt template in the conversational retrieval chain

    llm = ChatOpenAI()
    ## on the second thought-  lets just manage it outside
    ## We need to figure out the memory
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = vectorDb.as_retriever(),
        memory=memory,
        verbose=True,
        combine_docs_chain_kwargs={'prompt':QA_CHAIN_PROMPT})
    
    result = conversation_chain.invoke({"question":question})
    #result = result["answer"]

    return result

def conversationChainWmemory(vectorDb,question,chat_history):
    ## This is the function that gets the response 
    
    template = """Use the following pieces of context and Chat History to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
                {context}
                Chat History: {chat_history}
                Question: {question}
                Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","chat_history","question"],
                                    template=template,)
    ### We can pass in the prompt template in the conversational retrieval chain
    llm = ChatOpenAI()
    ## on the second thought-  lets just manage it outside
    ## We need to figure out the memory
    #memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = vectorDb.as_retriever(),
     #   memory=memory,
        verbose=False,
        combine_docs_chain_kwargs={'prompt':QA_CHAIN_PROMPT})
    
    result=  conversation_chain({"question":question,"chat_history":chat_history})
    return result
    




#create_local_db(filenames='Star_Wars_Book.pdf',collectionName="StarWars")

#vectorDb = retrieve_from_db("StarWars")
#print(vectorDb._collection.count())
#chat_history = []

#
#res = conversationChain(question, vectorDb)


#print(res["answer"])
#chat_history= [(question,res['answer'])]
#question2 = "What was my previous question?"
#res = conversationChain(question2, vectorDb)
#print(chat_history)
#print(res["answer"])

#chat_history =[]

#question = " How is Luke related to Anakin?"
#res1 = conversationChainWmemory(vectorDb,question,chat_history)
#print(res1["answer"])
#chat_history=[(question,res1["answer"])]
#question2 = "How were they separated?"
#res2 = conversationChainWmemory(vectorDb,question2,chat_history)
#print(res2["answer"])