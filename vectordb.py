import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

import pickle

from langchain_google_vertexai import VertexAIEmbeddings

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud import storage

# Initialize Vertex AI with your project-id and a location
PROJECT_ID = 'joon-sandbox'
LOCATION = "asia-southeast1" # @param {type:"string"}

vertexai.init(project=PROJECT_ID, location=LOCATION)

# Populate a variable named embedding_model with an instance of the
# langchain_google_vertexai class VertexAIEmbeddings.
embedding_model = VertexAIEmbeddings(model_name="text-embedding-004")

def load_txt_from_gcs(bucket_name, file_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    
    # Download the file content as a string
    content = blob.download_as_text(encoding="utf-8")
    return content

# Specify your GCS bucket name and file path
bucket_name = "alice-in-documentland"
file_name = "all_confluence_pages.txt"

# Load the text file from GCS
raw_data = load_txt_from_gcs(bucket_name, file_name)

# Use the embedding_model to generate embeddings of the text chunks, saving them to a list called chunked_embeddings.
chunked_embeddings = embedding_model.embed_documents(raw_data)

chunked_content = pickle.load(open("chunked_content.pkl", "rb"))
chunked_embeddings = pickle.load(open("chunked_embeddings.pkl", "rb"))


# Initializing the Firebase client
database_name = "joon-hackathon-chatbot"

# Set up the Firestore client
db = firestore.Client(project=PROJECT_ID, database=database_name)

# TODO: Instantiate a collection reference
collection = db.collection("confluence")

# Using a combination of our lists chunked_content and chunked_embeddings,
# add a document to your collection for each of your chunked documents.
for i, (content, embedding) in enumerate(zip(chunked_content, chunked_embeddings)):
    doc_ref = collection.document(f"doc_{i}")
    doc_ref.set({
        "content": content,
        "embedding": Vector(embedding)
    })

def search_vector_database(query: str):
    context = ""

    # 1. Generate the embedding of the query
    query_embedding = embedding_model.embed_query(query)

    # 2. Get the 5 nearest neighbors from your collection.
    vector_query = collection.find_nearest(
        vector_field="embedding",
        query_vector=Vector(query_embedding),
        distance_measure=DistanceMeasure.EUCLIDEAN,
        limit=5,
    )

    # 3. Call to_dict() on each snapshot to load its data.
    docs = vector_query.stream()
    context = [result.to_dict()['content'] for result in docs]

    return context

search_vector_database("Give me the latest blog")
