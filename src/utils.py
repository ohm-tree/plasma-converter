"""
Run 
```
export GOOGLE_APPLICATION_CREDENTIALS="/home/ubuntu/ohm-tree-filesys/plasma-converter-9e8e4fa06716.json"
source bashrc
```
in terminal beforehand.
"""

import os

from google.cloud import storage


def upload_to_gcs(source_file_name,
                  destination_blob_name,
                  bucket_name="plasma-experiments"):
    """Uploads a file to the Google Cloud Storage bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)

        print(
            f"File '{source_file_name}' successfully uploaded to '{destination_blob_name}' in bucket '{bucket_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}. Trying to upload as folder")
        upload_folder_to_gcs(
            source_file_name, destination_blob_name, bucket_name)


def upload_folder_to_gcs(source_folder_name, destination_folder_name, bucket_name="plasma-experiments"):
    """Uploads an entire folder to the Google Cloud Storage bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)

        # Walk through all files and subdirectories in the folder
        for root, _, files in os.walk(source_folder_name):
            for file_name in files:
                # Create the full local path of the file
                file_path = os.path.join(root, file_name)

                # Create a corresponding destination path in GCS
                relative_path = os.path.relpath(file_path, source_folder_name)
                destination_blob_name = os.path.join(
                    destination_folder_name, relative_path)

                # Upload the file
                blob = bucket.blob(destination_blob_name)
                blob.upload_from_filename(file_path)

                print(
                    f"File '{file_path}' successfully uploaded to '{destination_blob_name}' in bucket '{bucket_name}'.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    upload_to_gcs(
        "/home/ubuntu/ohm-tree-filesys/plasma-converter/logs/minif2f-valid-one-step_2024-10-18_19-25-54", "one_shot_results")
