import logging
import vertexai

from langchain_google_vertexai import VertexAIEmbeddings

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Vertex AI with your project-id and a location
PROJECT_ID = 'joon-sandbox'
LOCATION = "asia-southeast1"  # Ensure this matches your Firestore location
DATABASE_ID = 'joon-hackathon-chatbot'  # Specify the Firestore database ID

logging.info("Initializing Vertex AI")
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Initialize the embedding model
logging.info("Initializing the embedding model")
# embedding_model = VertexAIEmbeddings(model_name="text-embedding-004")
embedding_model = VertexAIEmbeddings(model_name="text-embedding-005")

def load_txt_from_gcs(bucket_name, file_name):
    logging.info(f"Loading text from GCS bucket: {bucket_name}, file: {file_name}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    
    # Download the file content as a string
    content = blob.download_as_text(encoding="utf-8")
    logging.info("Text loaded from GCS")
    return content

# Specify your GCS bucket name and file path
bucket_name = "alice-in-documentland"
file_name = "all_confluence_pages.txt"

# Load the text file from GCS
raw_data = load_txt_from_gcs(bucket_name, file_name)

# Demo: Limit to the first 1000 characters for quick testing
demo_data = raw_data[-100000:]  # Adjust this limit as needed for a quick demo
# demo_data = raw_data  # Adjust this limit as needed for a quick demo

# Use the embedding_model to generate embeddings of the demo text chunks
logging.info("Generating embeddings for the demo text chunks")
chunked_embeddings = embedding_model.embed_documents([demo_data])

# Initialize Firestore client
logging.info("Initializing Firestore client")
db = firestore.Client(project=PROJECT_ID, database=DATABASE_ID)

collection = db.collection("confluence")

# Store the demo document in Firestore
logging.info("Storing demo document in Firestore")
doc_ref = collection.document("demo_doc")
doc_ref.set({
    "content": demo_data,
    "embedding": Vector(chunked_embeddings[0])
})
logging.info("Demo document stored in Firestore")

def search_vector_database(query: str):
    logging.info(f"Searching vector database with query: {query}")
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

    logging.info("Search completed")
    print(context)
    return context

# Example search
search_vector_database("get the latest article")

