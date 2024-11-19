import subprocess
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

def load_documents(file_path):
    loader = TextLoader(file_path)
    return loader.load()

def split_documents(documents, chunk_size=500, chunk_overlap=50):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def create_faiss_vector_store(chunks, model_name):
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.from_documents(chunks, embedding_model)

def initialize_llama_model(model_name, temperature=0, num_predict=512):
    return ChatOllama(model=model_name, temperature=temperature, num_predict=num_predict)

def setup_retrieval_qa_chain(llm, vector_store):
    retriever = vector_store.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def chatbot_conversation(qa_chain):
    conversation_history = ""
    print("Chatbot ready! Type 'exit' to quit.")
    while True:
        user_input = input("Speak: ")
        if user_input.lower() == "bye":
            print("Goodbye!")
            break
        try:
            relevant_docs = qa_chain.retriever.get_relevant_documents(conversation_history + " " + user_input)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            response = qa_chain({"context": context, "query": user_input})
            print(f"Listen: {response['result']}")
            
            conversation_history += f"User: {user_input}\nBot: {response['result']}\n"
        except Exception as e:
            print(f"Error: {e}")

def main():
    document_path = "doc.txt"
    model_name = "llama3.2:1b"

    documents = load_documents(document_path)
    chunks = split_documents(documents)

    faiss_vector_store = create_faiss_vector_store(chunks, model_name)
    llm = initialize_llama_model(model_name)
    qa_chain = setup_retrieval_qa_chain(llm, faiss_vector_store)

    chatbot_conversation(qa_chain)

if __name__ == "__main__":
    main()
