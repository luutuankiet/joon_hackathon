from __future__ import annotations
import os
import streamlit as st
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from langchain_google_vertexai import VertexAIEmbeddings

# Application constants
BOTNAME = "Alice in Documentland"
SUBTITLE = "Your Friendly Document Expert"
PROJECT_ID = "joon-sandbox"
DATABASE_NAME = "test-db"
COLLECTION_NAME = "dbt_site_pages"

# Cache the initialization of Google Cloud services
@st.cache_resource
def initialize_firestore():
    """Initialize and cache Firestore client"""
    return firestore.Client(project=PROJECT_ID, database=DATABASE_NAME)

@st.cache_resource
def initialize_collection(_db):
    """Initialize and cache Firestore collection"""
    return db.collection(COLLECTION_NAME)

@st.cache_resource
def initialize_embedding_model():
    """Initialize and cache embedding model"""
    return VertexAIEmbeddings(model_name="text-embedding-005")

@st.cache_resource
def initialize_gen_model():
    """Initialize and cache generative model"""
    return GenerativeModel(
        model_name="gemini-pro",
        generation_config=GenerationConfig(temperature=0)
    )

# Initialize services with progress indicators
with st.spinner("Initializing services..."):
    db = initialize_firestore()
    collection = initialize_collection(db)
    embedding_model = initialize_embedding_model()
    gen_model = initialize_gen_model()

@st.cache_data(ttl=300)  # Cache results for 5 minutes
def search_vector_database(query: str):
    """Return relevant context from vector database with caching"""
    try:
        # Generate the embedding of the query
        query_embedding = embedding_model.embed_query(query)
        
        # Create a Vector object from the embedding values
        query_vector = Vector(query_embedding)
        
        # Get the 5 nearest neighbors
        vector_query = collection.find_nearest(
            vector_field="embedding_map",
            query_vector=query_vector,
            distance_measure=DistanceMeasure.EUCLIDEAN,
            limit=5,
        )
        
        # Process results
        docs = list(vector_query.stream())
        
        if not docs:
            return []
        
        # Extract content and metadata from documents
        context_entries = []
        for result in docs:
            doc_dict = result.to_dict()
            entry = {
                "title": doc_dict.get("title", "Untitled Document"),
                "url": doc_dict.get("url", ""),
                "content": doc_dict.get("content", ""),
                "summary": doc_dict.get("summary", ""),
                "chunk_number": doc_dict.get("chunk_number", "")
            }
            context_entries.append(entry)
        
    except Exception as e:
        st.error(f"Error in vector search: {e}")
        context_entries = []
    
    return context_entries

def ask_gemini(question):
    """Generate a response using Gemini with context from the vector database"""
    # Create a prompt template that instructs the model to separate answer from context
    prompt_template = (
        "You are an advanced AI assistant. Use the context provided below to answer the user's question "
        "as accurately as possible. Your response should be clear, concise, and directly address the question.\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Provide your answer in the following format:\n"
        "ANSWER: [Your detailed answer here, including relevant information from the context]\n\n"
        "SOURCES: [List the titles and URLs of the sources you used from the context, formatted as markdown links]"
    )

    # Retrieve context relevant to the question
    with st.spinner("Searching for relevant information..."):
        context_entries = search_vector_database(question)
    
    if not context_entries:
        return {
            "answer": "I couldn't find any relevant information to answer your question.",
            "context": [],
            "sources": "No sources found."
        }

    # Format context for the prompt
    formatted_context = "\n\n".join([
        f"Title: {entry['title']}\nURL: {entry['url']}\nSummary: {entry['summary']}\nContent: {entry['content']}"
        for entry in context_entries
    ])

    # Format the prompt template
    prompt = prompt_template.format(context=formatted_context, question=question)

    # Generate response with Gemini
    try:
        with st.spinner("Generating response..."):
            response = gen_model.generate_content(prompt)
            response_text = response.text
            
            # Parse the response to separate answer and sources
            parts = response_text.split("SOURCES:", 1)
            
            if len(parts) > 1:
                answer_part = parts[0].replace("ANSWER:", "").strip()
                sources_part = parts[1].strip()
            else:
                answer_part = response_text
                sources_part = "No specific sources cited."
            
            return {
                "answer": answer_part,
                "context": context_entries,
                "sources": sources_part
            }
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return {
            "answer": f"Error generating response: {str(e)}",
            "context": context_entries,
            "sources": "Error retrieving sources."
        }

def format_context_for_display(context_entries):
    """Format context entries for display in the UI"""
    if not context_entries:
        return "No context available."
    
    formatted_context = ""
    for i, entry in enumerate(context_entries):
        formatted_context += f"### Source {i+1}: {entry['title']}\n"
        formatted_context += f"**URL:** [{entry['url']}]({entry['url']})\n\n\n"
        formatted_context += f"**Scraped source summary:** {entry['summary']}\n\n"
    
    return formatted_context

def main():
    st.title(BOTNAME)
    st.write(SUBTITLE)

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": {
                    "answer": "Hi, I'm Alice in Documentland, what can I do for you?",
                    "context": [],
                    "sources": ""
                }
            }
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                # For user messages, just display the content directly
                st.markdown(message["content"])
            else:
                # For assistant messages, display the answer and collapsible context
                content = message["content"]
                
                # Display the answer
                st.markdown(content["answer"])
                
                # Display sources if available
                if content.get("sources") and content["sources"] != "No specific sources cited.":
                    st.markdown("**Sources:**")
                    st.markdown(content["sources"])
                
                # Add collapsible context section if context exists
                if content.get("context") and len(content["context"]) > 0:
                    with st.expander("View Reference Context"):
                        st.markdown(format_context_for_display(content["context"]))

    # Chat input for the user
    user_input = st.chat_input("What can I help you with?")

    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            # Get response from Gemini
            response = ask_gemini(user_input)
            
            # Update placeholder with response
            message_placeholder.empty()
            
            # Display the answer
            st.markdown(response["answer"])
            
            # Display sources if available
            if response.get("sources") and response["sources"] != "No specific sources cited.":
                st.markdown("**Sources:**")
                st.markdown(response["sources"])
            
            # Add collapsible context section if context exists
            if response.get("context") and len(response["context"]) > 0:
                with st.expander("View Reference Context"):
                    st.markdown(format_context_for_display(response["context"]))
            
            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response
            })

if __name__ == "__main__":
    main()
