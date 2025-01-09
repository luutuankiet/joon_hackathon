import vertexai
from vertexai.language_models import TextEmbeddingModel
from vertexai.generative_models import GenerativeModel

import pickle
from IPython.display import display, Markdown

from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

# Initialize Vertex AI with your project-id and a location
PROJECT_ID = ! gcloud config get-value project
PROJECT_ID = PROJECT_ID[0]
LOCATION = "us-central1" # @param {type:"string"}
print(PROJECT_ID)
vertexai.init(project=PROJECT_ID, location=LOCATION)

# Populate a variable named embedding_model with an instance of the
# langchain_google_vertexai class VertexAIEmbeddings.
from langchain_google_vertexai import VertexAIEmbeddings
embedding_model = VertexAIEmbeddings(model_name="text-embedding-004")

# https://python.langchain.com/v0.1/docs/modules/data_connection/document_loaders/pdf/#using-pymupdf
# Use the LangChain class PyMuPDFLoader to load the contents of the PDF
from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader("./nyc_food_safety_manual.pdf")
data = loader.load()

# Create a function to do some basic cleaning on artifacts found in this particular document.
def clean_page(page):
  return page.page_content.replace("-\n","")\
                          .replace("\n"," ")\
                          .replace("\x02","")\
                          .replace("\x03","")\
                          .replace("fo d P R O T E C T I O N  T R A I N I N G  M A N U A L","")\
                          .replace("N E W  Y O R K  C I T Y  D E P A R T M E N T  O F  H E A L T H  &  M E N T A L  H Y G I E N E","")

# Create a variable called cleaned_pages that is a list of strings, with each string being a page of content cleaned by above function.
cleaned_pages = []
for pages in data:
  cleaned_pages.append(clean_page(pages))


  docs = text_splitter.create_documents(cleaned_pages[0:4])
chunked_content = [doc.page_content for doc in docs]

# Use the embedding_model to generate embeddings of the text chunks, saving them to a list called chunked_embeddings.
# To do so, pass your list of chunks to the VertexAIEmbeddings class's embed_documents() method.
# https://python.langchain.com/v0.2/docs/integrations/text_embedding/google_vertex_ai_palm/
chunked_embeddings = embedding_model.embed_documents(chunked_content)



chunked_content = pickle.load(open("chunked_content.pkl", "rb"))
chunked_embeddings = pickle.load(open("chunked_embeddings.pkl", "rb"))


# Populate a db variable with a Firestore Client.
db = firestore.Client(project="qwiklabs-gcp-01-13a2450742aa")

collection = db.collection("food-safety")


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
  # Call the get() method on the result of your call to
  # find_nearest to retrieve document snapshots.
  vector_query = collection.find_nearest(
    vector_field="embedding",
    query_vector=Vector(query_embedding),
    distance_measure=DistanceMeasure.EUCLIDEAN,
    limit=5,
  )

  # 3. Call to_dict() on each snapshot to load its data.
  # Combine the snapshots into a single string named context
  docs = vector_query.stream()
  context = [result.to_dict()['content'] for result in docs]

  return context


search_vector_database("How should I store food?")