import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from llama_index.core.llms import ChatMessage
import logging
import time
from llama_index.llms.ollama import Ollama
from pymongo import MongoClient
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

logging.basicConfig(level=logging.INFO)


ollama_embedding = OllamaEmbeddings(
    model="all-minilm",
)
        

client = MongoClient("localhost", 27017)
database = client.ChatBotData
embeddings_collection = database.embeddings



if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'conversation' not in st.session_state:
    st.session_state.conversation = []


# Chat

def stream_chat(model, messages):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        resp = llm.stream_chat(messages)
        response = ""
        response_placeholder = st.empty()
        for r in resp:
            response += r.delta
            response_placeholder.write(response)
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e


# Embeddings

def generate_embeddings(question, response):
    try:
        logging.info("Generated embeddings")
        embeddings = ollama_embedding.embed_documents([question, response])
        embeddings_dict = {
            "request_embedding": embeddings[0],
            "response_embedding": embeddings[1],
        }

        vectorstore = FAISS.from_texts(texts=[question, response],  embedding=ollama_embedding)

        embeddings_data = {
            "question": question,
            "response": response,
            "request_embedding": embeddings_dict["request_embedding"],
            "response_embedding": embeddings_dict["response_embedding"],
            "timestamp": time.time(),
        }

        embeddings_collection.insert_one(embeddings_data)
        logging.info("Embeddings stored in MongoDB")

        return vectorstore
    except Exception as e:
        logging.error(f"Error while generating embeddings: {str(e)}")
        raise e



# Embedding PDF file

def generate_embeddings_pdf(chunks_text):
    if not chunks_text:
        logging.warning("No text found in uploaded PDFs")
        return

    try:
        logging.info("Generating Embedding for PDF file")
        embeddings = ollama_embedding.embed_documents([chunks_text])

        vectorstore = FAISS.from_texts(texts=chunks_text, embedding=ollama_embedding)

        embeddings_data = {
        "file": "PDF",
        "text": chunks_text,
        "embedding": embeddings,
        "timestamp": time.time(),
        }

        embeddings_collection.insert_one(embeddings_data)
        logging.info("Embeddings PDF file stored in MongoDB")


        return vectorstore
    except Exception as e:
        logging.error(f"Error while generating PDF file embeddings: {str(e)}")
        raise  e




# PDF

def get_pdf_text(pdf_documents):
    text = " "
    for pdf in pdf_documents:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


# PDF text chunks

def get_chunks_text(pdf_text):
    text_splitter = CharacterTextSplitter(separator='\n',
                                          chunk_size=1000,
                                          chunk_overlap=200,
                                          length_function=len
                                          )

    chunks = text_splitter.split_text(pdf_text)
    return chunks



# Conversation

def get_conversation_chain(vector_store):
    try:
        llm = Ollama(temperature=0.0, model="llama3.2:1b")

        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

        retriever = vector_store.as_retriever()

        conversation_chain = ConversationalRetrievalChain(llm=llm, retriever=retriever, memory=memory, combine_docs_chain=None, question_generator=None)

        return conversation_chain
    except Exception as e:
        logging.error(f"Error in get_conversation_chain: {str(e)}")
        raise e
    

# Main

def main():
    st.title("Chat with LLMs Models")
    logging.info("App started")
    model = st.sidebar.selectbox("Choose a model", ["llama3.2:1b"])
    logging.info(f"Model selected: {model}")

    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        logging.info(f"User input: {prompt}")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()
                logging.info("Generating response")

                with st.spinner("Writing..."):
                    try:
                        messages = [ChatMessage(role=msg["role"], content=msg["content"]) for msg in
                                    st.session_state.messages]
                        response_message = stream_chat(model, messages)
                        duration = time.time() - start_time

                        response_message_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response_message_with_duration})
                        st.write(f"Duration: {duration:.2f} seconds")
                        logging.info(f"Response: {response_message}, Duration: {duration:.2f} s")

                        embeddings = generate_embeddings(prompt, response_message)

                    except Exception as e:
                        st.session_state.messages.append({"role": "assistant", "content": str(e)})
                        st.error("An error occurred while generating the response.")
                        logging.error(f"Error: {str(e)}")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_documents = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                pdf_text = get_pdf_text(pdf_documents)

                chunks_text = get_chunks_text(pdf_text)

                vector_store = generate_embeddings_pdf(chunks_text)

                conversation = get_conversation_chain(vector_store)


# Run main function

if __name__ == "__main__":
    main()
