import os
from app_ import parse_csv_to_db
folder_path = "tweets/"

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        csv_file_path = os.path.join(folder_path, filename)
        print(f"Processing {csv_file_path} ...")
        try:
            tweets_added = parse_csv_to_db(csv_file_path)
            print(f"Added {tweets_added} tweets from {filename}")
        except Exception as e:
            print(f"Failed to process {filename}: {str(e)}")
