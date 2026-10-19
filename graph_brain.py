import os
import json
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from typing import TypedDict
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

load_dotenv()

with open("resume_data.json", "r") as f:
    resume_dict = json.load(f)

raw_text_list = []

for key,value in resume_dict.items():
    raw_text_list.append(f'{key}: {value}')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

resume_chunks = text_splitter.create_documents(raw_text_list)

embedding = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")


vector_db = FAISS.from_documents(documents=resume_chunks, embedding=embedding)
print("Vector database created successfully!")

#vector_db.save_local("faiss_index")
#print( "FAISS Database successfully created and saved!")

class GraphState(TypedDict):
    question: str
    context: str
    answer : str

llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0, max_tokens=800)

def router_node(state: GraphState):
    question = state["question"]
    print(f"Router checking question: {question}")
    
    prompt = PromptTemplate.from_template(
        "You are an intelligent routing assistant for Manthan's Resume Bot.\n"
        "If the user's question is about Manthan, his resume, skills, education, projects, OR any HR/Interview/Behavioral questions (e.g., 'why should we hire you', 'strengths', 'weaknesses', 'tell me about yourself', 'achievements'), you MUST output exactly 'VectorStore'.\n"
        "If the question is completely random, off-topic, or asks for general knowledge (e.g., 'what is python', 'write a poem', 'how to cook'), output exactly 'Fallback'.\n\n"
        "User Question: {question}\n"
        "Route:"
    )
    
    chain = prompt | llm | StrOutputParser()
    route = chain.invoke({"question": question})
    
    route = route.strip().replace("'", "").replace('"', '')
    if "VectorStore" in route:
        route = "VectorStore"
    elif "Fallback" in route:
        route = "Fallback"
    else:
        route = "Fallback" 
        
    print(f"Router decided route: {route}")
    return route

def retrieve_node(state: GraphState):
    question = state["question"]
    
    docs = vector_db.similarity_search(question)
    
    context = "\n".join([doc.page_content for doc in docs])
    
    print("🔍 Retriever found relevant context from FAISS.")
    
    return {"context": context}



def generator_node(state: GraphState):
    question = state["question"]
    context = state["context"]
    
 
    system_rules = resume_dict["System_Instructions"]
    
    
    prompt = PromptTemplate.from_template(
        "You are Manthan's professional AI Assistant. You must strictly follow these rules:\n"
        "{rules}\n\n"
        "Here is the relevant information from Manthan's resume:\n"
        "{context}\n\n"
        "User Question: {question}\n"
        "Answer:"
    )
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"rules": system_rules, "context": context, "question": question}) 
    
    print("Generator created the answer!")
    return {"answer": answer}


def outoff_context_node(state: GraphState):

    print("Fallback Node triggered: Out of context question.")

    fallback_message = (

        "I am an AI assistant strictly designed to answer questions about Manthan's resume, "
        "skills, and professional experience. I cannot answer outside queries. "
        "For personal details or other topics, please contact Manthan directly at manthankhonde@gmail.com."
   
    )

    return{"answer":fallback_message}

workflow = StateGraph(GraphState)


workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("generator_node", generator_node)
workflow.add_node("outoff_context_node", outoff_context_node)

workflow.add_conditional_edges(

    START,

    router_node,
    {
        "VectorStore": "retrieve_node",
        "Fallback": "outoff_context_node"
    }
)

workflow.add_edge("retrieve_node", "generator_node")

workflow.add_edge("generator_node", END)

workflow.add_edge("outoff_context_node", END)

app = workflow.compile()


