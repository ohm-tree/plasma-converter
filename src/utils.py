# Run `export GOOGLE_APPLICATION_CREDENTIALS="/home/ubuntu/ohm-tree-filesys-tokyo/plasma-converter-9e8e4fa06716.json"` 
# source bashrc

from google.cloud import storage
import os

def upload_to_gcs(source_file_name,
                  destination_blob_name,
                  bucket_name="plasma-experiments"):
    """Uploads a file to the Google Cloud Storage bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)

        print(f"File '{source_file_name}' successfully uploaded to '{destination_blob_name}' in bucket '{bucket_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")