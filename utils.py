import os
import requests
import zipfile
from pathlib import Path


class Helpers:
    def download_dataset():
        record_id = "5171712"
        save_dir = Path(__file__).parent / "Data"
        save_dir.mkdir(exist_ok=True)
    
        required_folders = {"PV03", "PV01", "PV08"}
    
        # Determine what's already present
        existing_folders = {f.name for f in save_dir.iterdir() if f.is_dir()}
        existing_zips = {f.stem for f in save_dir.glob("*.zip")}
        missing = required_folders - (existing_folders | existing_zips)
    
        # Extract ZIPs if folder is missing
        for zip_file in save_dir.glob("*.zip"):
            base = zip_file.stem
            if base in required_folders and base not in existing_folders:
                print(f"Extracting existing ZIP: {zip_file.name}")
                try:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(save_dir / base)
                    zip_file.unlink()
                    print(f"Extracted and deleted: {zip_file.name}")
                except zipfile.BadZipFile:
                    print(f"Corrupted ZIP skipped: {zip_file.name}")
    
        # Download missing ZIPs
        if not missing:
            print("All required data present.")
            return
    
        print(f"Downloading missing: {missing}")
        metadata_url = f"https://zenodo.org/api/records/{record_id}"
        response = requests.get(metadata_url)
        data = response.json()
    
        for file in data['files']:
            base = file['key'].split(".")[0]
            if base in missing and file['key'].endswith(".zip"):
                url = file['links']['self']
                zip_path = save_dir / file['key']
                print(f"Downloading {file['key']}...")
                r = requests.get(url)
                with open(zip_path, 'wb') as f:
                    f.write(r.content)
    
                # Extract and delete
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(save_dir / base)
                    zip_path.unlink()
                    print(f"Extracted and deleted: {file['key']}")
                except zipfile.BadZipFile:
                    print(f"Failed to extract: {file['key']}")
        print("Dataset setup complete.")
    
