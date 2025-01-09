import os
from dotenv import load_dotenv
import requests
import pandas as pd
from bs4 import BeautifulSoup
from google.cloud import storage
import tempfile
from datetime import datetime



load_dotenv()

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# Environment variables
CONFLUENCE_URL = os.getenv('CONFLUENCE_URL')
CONFLUENCE_SPACE_NUMBER = os.getenv('CONFLUENCE_SPACE_NUMBER')
ATLASSIAN_TOKEN_SECRET_NAME = os.getenv('ATLASSIAN_TOKEN_SECRET_NAME')
ATLASSIAN_EMAIL_SECRET_NAME = os.getenv('ATLASSIAN_EMAIL_SECRET_NAME')
ATLASSIAN_TOKEN_SECRET_VALUE = os.getenv('ATLASSIAN_TOKEN_SECRET_VALUE')
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')

class ConfluenceDataLoader():
    def __init__(self):
        self.CONFLUENCE_URL = CONFLUENCE_URL
        self.CONFLUENCE_SPACE_NUMBER = CONFLUENCE_SPACE_NUMBER
        self.ATLASSIAN_TOKEN_SECRET_NAME = ATLASSIAN_TOKEN_SECRET_NAME
        self.ATLASSIAN_EMAIL_SECRET_NAME = ATLASSIAN_EMAIL_SECRET_NAME
        self.ATLASSIAN_TOKEN_SECRET_VALUE = ATLASSIAN_TOKEN_SECRET_VALUE
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(GCS_BUCKET_NAME)

    # Function to get page's labels based on page_id
    def fetch_page_labels(self, page_id: str, auth: any):
        labels_url = f"{self.CONFLUENCE_URL}/wiki/api/v2/pages/{page_id}/labels"
        headers = {"Accept": "application/json"}
        response = requests.get(labels_url, headers=headers, auth=auth)
        if response.status_code == 200:
            labels_data = response.json()
            labels = [label['name'] for label in labels_data['results']]
            return ', '.join(labels)
        else:
            print(f"Failed to retrieve labels for page ID {page_id}: {response.text}")
            return ''

        # Function to get all pages' details
    def get_all_pages(self, auth: any):
        api_url = f"{self.CONFLUENCE_URL}/wiki/api/v2/spaces/{self.CONFLUENCE_SPACE_NUMBER}/pages"
        params = {
            "limit": 250,  # The max limit for pages' contents of each api call
            "body-format": "storage"
        }
        headers = {"Accept": "application/json"}
        all_pages_details = []
        while True:
            response = requests.get(api_url, headers=headers, params=params, auth=auth)
            if response.status_code != 200:
                print(f"Failed to retrieve pages: {response.text}")
                break
            data = response.json()
            pages = data.get("results", [])
            for page in pages:
                page_id = page.get("id", "")
                title = page.get("title", "")
                createdAt = page.get("createdAt", "")
                body = page.get("body", {}).get("storage", {}).get("value", "")
                soup = BeautifulSoup(body, 'html.parser')
                text_content = soup.get_text().strip()
                page_url = f"{CONFLUENCE_URL}/wiki/spaces/RF/pages/{page_id}/{title.replace(' ', '+')}"
                print(f"working on page: {title}")
                if text_content:
                    labels = self.fetch_page_labels(page_id, auth)
                    page_details = {
                        "page_id": page_id,
                        "title": title,
                        "content": text_content,
                        "labels": labels,
                        "created_at": createdAt,
                        "url": page_url
                    }
                    all_pages_details.append(page_details)
            
            if "next" in data.get("_links", {}):
                api_url = self.CONFLUENCE_URL + data["_links"]["next"]
            else:
                break

        print(f"Total pages fetched: {len(all_pages_details)}")
        return all_pages_details

    def save_all_to_text(self, all_contents: list) -> str:
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as tmp:
            for content in all_contents:
                tmp.write(f"Title: {content['title']}\n")
                tmp.write(f"Created: {content['created_at']}\n")
                tmp.write(f"Labels: {content['labels']}\n")
                tmp.write(f"URL: {content['url']}\n\n")
                tmp.write(content['content'])
                tmp.write("\n\n" + "="*80 + "\n\n")  # Separator between pages
            return tmp.name



    def clear_bucket(self):
        blobs = self.bucket.list_blobs()
        for blob in blobs:
            blob.delete()
        print(f"All objects in bucket {GCS_BUCKET_NAME} have been deleted.")

    def upload_to_gcs(self, file_path: str, file_name: str) -> str:
        blob = self.bucket.blob(file_name)
        blob.upload_from_filename(file_path)
        return f"gs://{GCS_BUCKET_NAME}/{file_name}"

    def run(self):
        auth = requests.auth.HTTPBasicAuth(self.ATLASSIAN_EMAIL_SECRET_NAME, self.ATLASSIAN_TOKEN_SECRET_VALUE)
        all_pages = self.get_all_pages(auth)
        page_contents = [page for page in all_pages if page]
        print(f"Total pages found: {len(page_contents)}")

        if page_contents:
            self.clear_bucket()
            text_path = self.save_all_to_text(page_contents)
            gcs_path = self.upload_to_gcs(text_path, "all_confluence_pages.txt")
            os.unlink(text_path)

            print(f'Uploaded combined text file to GCS: {gcs_path}')
            return gcs_path
        else:
            print("No pages with content were found.")
            return None

if __name__ == "__main__":
    loader = ConfluenceDataLoader()
    uploaded_file = loader.run()
