import os
from dotenv import load_dotenv
import requests
import pandas as pd
from bs4 import BeautifulSoup
from google.cloud import secretmanager
from google.cloud import storage
import tempfile
import pdfkit
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
SECRET_PROJECT_ID = os.getenv('SECRET_PROJECT_ID')
GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME')

class ConfluenceDataLoader():
    def __init__(self):
        self.CONFLUENCE_URL = CONFLUENCE_URL
        self.CONFLUENCE_SPACE_NUMBER = CONFLUENCE_SPACE_NUMBER
        self.ATLASSIAN_TOKEN_SECRET_NAME = ATLASSIAN_TOKEN_SECRET_NAME
        self.ATLASSIAN_EMAIL_SECRET_NAME = ATLASSIAN_EMAIL_SECRET_NAME
        self.SECRET_PROJECT_ID = SECRET_PROJECT_ID
        self.ATLASSIAN_EMAIL = self.read_gcp_secret_value(ATLASSIAN_EMAIL_SECRET_NAME)
        self.ATLASSIAN_TOKEN = self.read_gcp_secret_value(ATLASSIAN_TOKEN_SECRET_NAME)
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(GCS_BUCKET_NAME)

    # Function to retrieve value from GCP Secret Manager:
    def read_gcp_secret_value(self, secret_name: str):
        client = secretmanager.SecretManagerServiceClient()
        secret_version = f"projects/{self.SECRET_PROJECT_ID}/secrets/{secret_name}/versions/latest"
        response = client.access_secret_version(request={"name": secret_version})
        return response.payload.data.decode("UTF-8")

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
        while True:
            response = requests.get(api_url, headers=headers, params=params, auth=auth)
            if response.status_code != 200:
                print(f"Failed to retrieve pages: {response.text}")
                break
            data = response.json()
            pages = data.get("results", [])
            all_pages_details = []
            for page in pages:
                page_id = page.get("id", "")
                title = page.get("title", "")
                createdAt = page.get("createdAt", "")
                body = page.get("body", {}).get("storage", {}).get("value", "")
                soup = BeautifulSoup(body, 'html.parser')
                text_content = soup.get_text().strip()
                page_url = f"{CONFLUENCE_URL}/wiki/spaces/RF/pages/{page_id}/{title.replace(' ', '+')}"
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
        return all_pages_details

    def save_to_pdf(self, content: dict) -> str:
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            html_content = f"""
            <h1>{content['title']}</h1>
            <p>Created: {content['created_at']}</p>
            <p>Labels: {content['labels']}</p>
            <p>URL: {content['url']}</p>
            <div>{content['content']}</div>
            """
            pdfkit.from_string(html_content, tmp.name)
            return tmp.name

    def upload_to_gcs(self, pdf_path: str, content: dict) -> str:
        today = datetime.now().strftime('%Y/%m/%d')
        blob_name = f"confluence_docs/{today}/{content['page_id']}.pdf"
        blob = self.bucket.blob(blob_name)
        blob.upload_from_filename(pdf_path)
        return f"gs://{GCS_BUCKET_NAME}/{blob_name}"

    def run(self):
        auth = requests.auth.HTTPBasicAuth(self.ATLASSIAN_EMAIL, self.ATLASSIAN_TOKEN)
        all_pages = self.get_all_pages(auth)
        page_contents = [page for page in all_pages if page]
        print(f"Total pages found: {len(page_contents)}")

        uploaded_files = []
        if page_contents:
            for content in page_contents:
                pdf_path = self.save_to_pdf(content)
                gcs_path = self.upload_to_gcs(pdf_path, content)
                uploaded_files.append(gcs_path)
                os.unlink(pdf_path)

            print(f'Uploaded {len(uploaded_files)} PDFs to GCS')
            return uploaded_files
        else:
            print("No pages with content were found.")
            return []

if __name__ == "__main__":
    loader = ConfluenceDataLoader()
    uploaded_files = loader.run()
