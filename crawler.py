import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

# Google Cloud imports
from google.cloud import firestore
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.language_models import TextEmbeddingModel
from google.cloud.firestore_v1.vector import Vector
load_dotenv()

# Initialize Firestore client
# credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
# if credentials_path and os.path.exists(credentials_path):
#     credentials = service_account.Credentials.from_service_account_file(credentials_path)
#     db = firestore.Client(credentials=credentials)
# else:
    # Use default credentials
    # db = firestore.Client()
db = firestore.Client(database='test-db')

# Initialize Vertex AI
project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
location = os.getenv("VERTEX_LOCATION", "us-central1")
vertexai.init(project=project_id, location=location)

# Initialize Gemini model
gemini_model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
gemini_model = GenerativeModel(gemini_model_name)

# Initialize embedding model
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
# embedding_model = VertexAIEmbeddings(model_name="text-embedding-004")

# Define the Crawl4AI API endpoint (Docker service)
CRAWL4AI_API_URL = "http://localhost:11235"
CRAWL4AI_API_TOKEN = os.getenv("CRAWL4AI_API_TOKEN", "")

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first ()
        chunk = text[start:end]
        code_block = chunk.rfind('')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

import time
import random
from asyncio import Semaphore, Lock

# Create a rate limiter for Gemini requests
# Default quota for Gemini 1.5 Pro is 60 requests per minute per project
# We'll set it to 40 to be safe
MAX_REQUESTS_PER_MINUTE = 40
gemini_lock = Lock()  # Use a lock for precise control
gemini_requests = []  # Track timestamps of requests

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using Gemini with strict rate limiting."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""
    
    # Implement strict rate limiting
    async with gemini_lock:
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        global gemini_requests
        gemini_requests = [t for t in gemini_requests if current_time - t < 60]
        
        # If we're at or near the limit, wait until we're under the limit
        if len(gemini_requests) >= MAX_REQUESTS_PER_MINUTE:
            # Calculate how long to wait - time until oldest request is 60 seconds old plus buffer
            wait_time = 60 - (current_time - gemini_requests[0]) + 1.0  # Add a 1 second buffer
            print(f"Rate limit reached ({len(gemini_requests)}/{MAX_REQUESTS_PER_MINUTE}), waiting {wait_time:.2f} seconds before next Gemini request")
            await asyncio.sleep(wait_time)
            
            # Recalculate after waiting
            current_time = time.time()
            gemini_requests = [t for t in gemini_requests if current_time - t < 60]
    
    # Try to get title and summary with retries
    max_retries = 3
    for retry in range(max_retries):
        try:
            # Add a small random delay to spread out requests
            await asyncio.sleep(random.uniform(0.1, 0.5))
            
            # Record this request timestamp before making the request
            async with gemini_lock:
                gemini_requests.append(time.time())
                current_count = len(gemini_requests)
            
            # Create a synchronous function to call Gemini
            def call_gemini():
                generation_config = GenerationConfig(
                    temperature=0.2,
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=1024,
                    response_mime_type="application/json"
                )
                
                prompt = f"{system_prompt}\n\nURL: {url}\n\nContent:\n{chunk[:1000]}..."
                response = gemini_model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                return response.text
            
            # Run the synchronous function in a thread pool
            loop = asyncio.get_running_loop()
            response_text = await loop.run_in_executor(None, call_gemini)
            
            # Parse the JSON response
            try:
                # Try to parse the entire response as JSON
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # If that fails, try to extract JSON from the text
                import re
                json_match = re.search(r'({.*})', response_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group(1))
                else:
                    # If all else fails, create a basic response
                    result = {
                        "title": "Untitled Document Section",
                        "summary": "Summary extraction failed"
                    }
            
            # Ensure the required keys are present
            if "title" not in result or "summary" not in result:
                result = {
                    "title": result.get("title", "Untitled Document Section"),
                    "summary": result.get("summary", "Summary extraction failed")
                }
                
            return result
            
        except Exception as e:
            print(f"Error getting title and summary with Gemini (attempt {retry+1}/{max_retries}): {e}")
            
            # If it's a rate limit error, wait longer
            if "429" in str(e) or "Quota exceeded" in str(e):
                # Exponential backoff with jitter for rate limit errors
                backoff_time = (2 ** retry) * 5 + random.uniform(1, 5)
                print(f"Rate limit exceeded. Backing off for {backoff_time:.2f} seconds")
                await asyncio.sleep(backoff_time)
            else:
                # For other errors, shorter backoff
                await asyncio.sleep(random.uniform(1, 3))
                
            # If this was the last retry, return a default response
            if retry == max_retries - 1:
                return {
                    "title": f"Document Section from {url.split('/')[-1]}",
                    "summary": "Summary unavailable due to API limits"
                }

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from Vertex AI."""
    try:
        # Create a synchronous function to call the embedding model
        def call_embedding_model():
            embeddings = embedding_model.get_embeddings([text])
            # embeddings = embedding_model.embed_documents([text])
            # return embeddings
            return embeddings[0].values
        
        # Run the synchronous function in a thread pool
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, call_embedding_model)
        
        return embedding
    except Exception as e:
        print(f"Error getting embedding from Vertex AI: {e}")
        # Return a zero vector with the same dimension as the model's output
        # Gecko model typically returns 768-dimensional embeddings
        return [0.0] * 768

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": "pydantic_ai_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Firestore."""
    try:
        # Create a document ID based on URL and chunk number
        safe_url = chunk.url.replace("/", "_").replace(".", "_")
        doc_id = f"{safe_url}_{chunk.chunk_number}"
        
        # Convert the chunk to a dictionary for Firestore
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            # Store embedding as a map of indices to values
            "embedding_map": Vector(chunk.embedding),
            # "embedding_map": {str(i): v for i, v in enumerate(chunk.embedding)},
            "created_at": firestore.SERVER_TIMESTAMP
        }
        
        # Add to Firestore
        collection_name = "dbt_site_pages"
        doc_ref = db.collection(collection_name).document(doc_id)
        doc_ref.set(data)
        
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return doc_id
    except Exception as e:
        print(f"Error inserting chunk into Firestore: {e}")
        return None

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel with strict rate limiting."""
    # Split into chunks
    chunks = chunk_text(markdown)
    print(f"Processing document with {len(chunks)} chunks: {url}")
    
    # Process chunks in very small batches to avoid overwhelming the rate limiter
    batch_size = 5  # Process just 5 chunks at a time
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        # Process current batch in parallel
        tasks = [
            process_chunk(chunk, i + j, url) 
            for j, chunk in enumerate(batch)
        ]
        processed_chunks = await asyncio.gather(*tasks)
        
        # Store processed chunks
        insert_tasks = [
            insert_chunk(chunk) 
            for chunk in processed_chunks
        ]
        await asyncio.gather(*insert_tasks)
        
        # Add a delay between batches to respect rate limits
        if i + batch_size < len(chunks):
            batch_num = i//batch_size + 1
            total_batches = (len(chunks) + batch_size - 1)//batch_size
            print(f"Processed batch {batch_num}/{total_batches} for {url}, waiting before next batch...")
            # Wait longer between batches to stay well under the limit
            await asyncio.sleep(5)  # 5 second delay between batches

async def crawl_parallel(urls: List[str], max_concurrent: int = 10):
    """Crawl multiple URLs in parallel with a concurrency limit using the Crawl4AI Docker API."""
    # Create a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Setup headers for API authentication if token is provided
    headers = {"Authorization": f"Bearer {CRAWL4AI_API_TOKEN}"} if CRAWL4AI_API_TOKEN else {}
    
    async def process_url(url: str):
        async with semaphore:
            # Prepare the request payload for the Docker API
            payload = {
                "urls": url,
                "priority": 10,
                "crawler_params": {
                    "headless": True,
                    "verbose": False
                },
                "extra": {
                    "bypass_cache": True  # Equivalent to CacheMode.BYPASS
                }
            }
            
            try:
                # Submit the crawl job to the Docker API
                response = requests.post(
                    f"{CRAWL4AI_API_URL}/crawl",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                task_id = response.json()["task_id"]
                print(f"Submitted task for URL: {url}, Task ID: {task_id}")
                
                # Poll for the result
                while True:
                    result_response = requests.get(
                        f"{CRAWL4AI_API_URL}/task/{task_id}",
                        headers=headers
                    )
                    result_response.raise_for_status()
                    status = result_response.json()
                    
                    if status["status"] == "completed":
                        if status.get("result", {}).get("success", False):
                            print(f"Successfully crawled: {url}")
                            # Extract the markdown content
                            markdown = status["result"]["markdown"]
                            await process_and_store_document(url, markdown)
                        else:
                            print(f"Failed: {url} - Error: {status.get('result', {}).get('error_message', 'Unknown error')}")
                        break
                    elif status["status"] == "failed":
                        print(f"Failed: {url} - Task failed")
                        break
                    
                    # Wait before polling again
                    await asyncio.sleep(10)
                    
            except Exception as e:
                print(f"Error processing URL {url}: {e}")
    
    # Process all URLs in parallel with limited concurrency
    await asyncio.gather(*[process_url(url) for url in urls])

def get_pydantic_ai_docs_urls() -> List[str]:
    """Get URLs from Pydantic AI docs sitemap."""
    sitemap_url = "https://docs.getdbt.com/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # Parse the XML
        root = ElementTree.fromstring(response.content)
        
        # Extract all URLs from the sitemap
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def main():
    # Get URLs from Pydantic AI docs
    urls = get_pydantic_ai_docs_urls()
    if not urls:
        print("No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())
