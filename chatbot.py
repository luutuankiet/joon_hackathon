import os
import yaml
import logging
import google.cloud.logging
from flask import Flask, render_template, request

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from langchain_google_vertexai import VertexAIEmbeddings

# Configure Cloud Logging
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)

# Read application variables from the config fle
BOTNAME = "Alice in Documentland"
SUBTITLE = "Your Friendly Document Expert"

app = Flask(__name__)

# Initializing the Firebase client
project_id = "joon-sandbox"
# database_name = "joon-hackathon-chatbot"
database_name = "test-db"

# Set up the Firestore client
db = firestore.Client(project=project_id, database=database_name)

# TODO: Instantiate a collection reference
collection = db.collection("dbt_site_pages")
# collection = db.collection("confluence")

# TODO: Instantiate an embedding model here
embedding_model = VertexAIEmbeddings(model_name="text-embedding-005")

# TODO: Instantiate a Generative AI model here
gen_model = model = GenerativeModel(
    model_name="gemini-pro",
    generation_config=GenerationConfig(temperature=0))

# TODO: Implement this function to return relevant context
# from your vector database
def search_vector_database(query: str):
    # For debugging - check if collection has documents
    all_docs = list(collection.limit(3).stream())
    print(f"Collection has {len(all_docs)} documents")
    if not all_docs:
        return ["No documents found in the collection."]
    
    # 1. Generate the embedding of the query using the same model as data loading
    try:
        
        query_embedding = embedding_model.embed_query(query)
        
        # Create a Vector object from the embedding values
        query_vector = Vector(query_embedding)
        
        # 2. Get the 5 nearest neighbors
        vector_query = collection.find_nearest(
            vector_field="embedding_map",
            query_vector=query_vector,  # Use Vector object, not dictionary
            distance_measure=DistanceMeasure.EUCLIDEAN,
            limit=5,
        )
        # vector_query = collection.find_nearest(
        #     vector_field="embedding_map",
        #     query_vector=query_vector,  # Use Vector object, not dictionary
        #     distance_measure=DistanceMeasure.EUCLIDEAN,
        #     limit=5,
        # )
        
        # 3. Process results
        docs = list(vector_query.stream())  # Convert to list to avoid stream consumption issues
        print(f"Vector query returned {len(docs)} documents")
        
        if not docs:
            return ["No relevant documents found for your query."]
        
        # Extract content from documents
        context = [result.to_dict().get('content', '') for result in docs]
        context_str = "\n\n".join(context)
        
    except Exception as e:
        print(f"Error in vector search: {e}")
        context_str = f"Error performing vector search: {str(e)}"
    
    # Don't delete this logging statement.
    logging.info(
        context_str, extra={"labels": {"service": "joon-service", "component": "context"}}
    )
    return context_str

# TODO: Implement this function to pass Gemini the context data,
# generate a response, and return the response text.
def ask_gemini(question):

    # 1. Create a prompt_template with instructions to the model
    # to use provided context info to answer the question.
    prompt_template = (
        "You are an advanced AI assistant. Use the context provided below to answer the user's question "
        "as accurately as possible. If no relevant context is available, say so explicitly.\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )

    # 2. Use your search_vector_database function to retrieve context
    # relevant to the question.
    context = search_vector_database(question)

    # 3. Format the prompt template with the question & context
    prompt = prompt_template.format(context=context, question=question)

    # 4. Pass the complete prompt template to gemini and get the text
    # of its response to return below.
    try:
        response = gen_model.generate_content(prompt)
        print(response.text)
    except Exception as e:
        response = f"{e}"

# The Home page route
@app.route("/", methods=["POST", "GET"])
def main():

    # The user clicked on a link to the Home page
    # They haven't yet submitted the form
    if request.method == "GET":
        question = ""
        answer = "Hi, I'm Alice in Documentland, what can I do for you?"

    # The user asked a question and submitted the form
    # The request.method would equal 'POST'
    else:
        question = request.form["input"]
        # Do not delete this logging statement.
        logging.info(
            question,
            extra={"labels": {"service": "joon-service", "component": "question"}},
        )
        
        # Ask Gemini to answer the question using the data
        # from the database
        answer = ask_gemini(question)

    # Do not delete this logging statement.
    logging.info(
        answer, extra={"labels": {"service": "joon-service", "component": "answer"}}
    )
    print("Answer: " + answer)

    # Display the home page with the required variables set
    config = {
        "title": BOTNAME,
        "subtitle": SUBTITLE,
        "botname": BOTNAME,
        "message": answer,
        "input": question,
    }

    return render_template("index.html", config=config)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 3306)))
