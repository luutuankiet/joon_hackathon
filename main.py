import requests
import pandas as pd
from bs4 import BeautifulSoup
from google.cloud import secretmanager

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python_operator import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryCreateEmptyTableOperator,
    BigQueryDeleteTableOperator,
    BigQueryValueCheckOperator
)
from google.cloud import bigquery
from airflow.models.param import Param
from common.common_var import (
    IS_DEV
)
from pendulum import duration, datetime
from datetime import timedelta

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

# Replace these values with your Confluence URL and the name of your secret in Secret Manager
CONFLUENCE_URL = "https://restoreportal.atlassian.net"
# The number id of Confluence Restore POW space
# To do: If there is more than one page we need to extract data from, we could use space_id API endpoint
CONFLUENCE_SPACE_NUMBER = "499155012"
ATLASSIAN_TOKEN_SECRET_NAME = "atlassian_api_token"  # The name of your atlassian token secret in Secret Manager
ATLASSIAN_EMAIL_SECRET_NAME = "atlassian_email"  # The name of your atlassian token email in Secret Manager
SECRET_PROJECT_ID = "restore-bi-dev-orchestration" if IS_DEV else "restore-bi-prod-orchestration"

DEFAULT_DESTINATION_PROJECT_ID = "restore-bi-dev-data-lake" if IS_DEV else "restore-bi-prod-data-lake"
DEFAULT_DESTINATION_DATASET_ID = "rf_confluence"
DEFAULT_DESTINATION_TABLE_ID = "rf_confluence_content"

GCP_CONN_ID = "gcloud_dev_data_lake" if IS_DEV else "gcloud_prod_data_lake"


class ConfluenceDataLoader():
    def __init__(self):
        self.CONFLUENCE_URL = CONFLUENCE_URL
        self.CONFLUENCE_SPACE_NUMBER = CONFLUENCE_SPACE_NUMBER
        self.ATLASSIAN_TOKEN_SECRET_NAME = ATLASSIAN_TOKEN_SECRET_NAME
        self.ATLASSIAN_EMAIL_SECRET_NAME = ATLASSIAN_EMAIL_SECRET_NAME
        self.SECRET_PROJECT_ID = SECRET_PROJECT_ID
        self.GCP_CONN_ID = GCP_CONN_ID
        self.ATLASSIAN_EMAIL = self.read_gcp_secret_value(ATLASSIAN_EMAIL_SECRET_NAME)
        self.ATLASSIAN_TOKEN = self.read_gcp_secret_value(ATLASSIAN_TOKEN_SECRET_NAME)

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

    # Default function for the class to transform and load all necessary data to BigQuery destination table
    def __call__(self, **kwargs):
        DESTINATION_PROJECT_ID = kwargs["destination_project_id"]
        DESTINATION_DATASET_ID = kwargs["destination_dataset_id"]
        DESTINATION_TABLE_ID = kwargs["destination_table_id"]

        auth = requests.auth.HTTPBasicAuth(self.ATLASSIAN_EMAIL, self.ATLASSIAN_TOKEN)
        all_pages = self.get_all_pages(auth)
        page_contents = [page for page in all_pages if page]
        print(f"Total pages found: {len(page_contents)}")

        if page_contents:

            df_page_contents = pd.DataFrame(page_contents)
            client = bigquery.Client()
            table_ref = f"{DESTINATION_PROJECT_ID}.{DESTINATION_DATASET_ID}.{DESTINATION_TABLE_ID}"
            client.load_table_from_dataframe(df_page_contents, table_ref)
            print(f'Loaded {len(df_page_contents)} rows into {DESTINATION_TABLE_ID}')

            kwargs['ti'].xcom_push(key='merged_df_length', value=len(df_page_contents))
        else:
            print("No pages with content were found.")


default_args = {
    "owner": "nguyendnb",
    "depends_on_past": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": duration(minutes=5),
    "start_date": datetime(2024, 1, 1),
    "params": {
        "destination_project_id": Param(
            default=DEFAULT_DESTINATION_PROJECT_ID,
            description="Destination project id to ingest data into",
            type="string",
        ),
        "destination_dataset_id": Param(
            default=DEFAULT_DESTINATION_DATASET_ID,
            description="Destination dataset id to ingest data into",
            type="string",
        ),
        "destination_table_id": Param(
            default=DEFAULT_DESTINATION_TABLE_ID,
            description="Destination table id to ingest data into",
            type="string",
        )
    }
}

with DAG(
    'ingest__confluence_to_bigquery',
    default_args=default_args,
    description='DAG to snapshot BigQuery tables',
    schedule_interval=timedelta(days=1),
    catchup=False
) as dag:

    start_task = EmptyOperator(task_id='start', dag=dag)
    end_task = EmptyOperator(task_id='end', dag=dag)

    deleting_existing_table = BigQueryDeleteTableOperator(
        task_id='deleting_existing_table',
        deletion_dataset_table="""{{ params.destination_project_id }}.{{ params.destination_dataset_id }}.{{ params.destination_table_id }}""",
        ignore_if_missing=True,
        gcp_conn_id=GCP_CONN_ID
    )

    creating_new_table = BigQueryCreateEmptyTableOperator(
        task_id='creating_new_empty_table',
        project_id="{{ params.destination_project_id }}",
        dataset_id="{{ params.destination_dataset_id }}",
        table_id="{{ params.destination_table_id }}",
        schema_fields=[
            {'name': 'page_id', 'type': 'string'},
            {'name': 'title', 'type': 'string'},
            {'name': 'content', 'type': 'string'},
            {'name': 'labels', 'type': 'string'},
            {'name': 'created_at', 'type': 'string'},
            {'name': 'url', 'type': 'string'}
        ],
        gcp_conn_id=GCP_CONN_ID
    )

    loading_dataframe = PythonOperator(
        task_id='loading_dataframe_to_bigquery',
        python_callable=ConfluenceDataLoader(),
        op_kwargs={
            "destination_project_id": "{{ params.destination_project_id }}",
            "destination_dataset_id": "{{ params.destination_dataset_id }}",
            "destination_table_id": "{{ params.destination_table_id }}",
        },
        provide_context=True
    )

    validating_loaded_rows = BigQueryValueCheckOperator(
        task_id="validating_loaded_rows",
        sql="SELECT COUNT(*) FROM {{ params.destination_project_id }}.{{ params.destination_dataset_id }}.{{ params.destination_table_id }}",
        pass_value="{{ task_instance.xcom_pull(task_ids='loading_dataframe_to_bigquery', key='merged_df_length') }}",
        use_legacy_sql=False,
        gcp_conn_id=GCP_CONN_ID
    )

    start_task >> deleting_existing_table >> creating_new_table >> loading_dataframe >> validating_loaded_rows >> end_task
