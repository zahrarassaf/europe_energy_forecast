import requests
import os
import pandas as pd

def download_real_dataset():
    """Download your real dataset from Google Drive"""
    data_path = "data/europe_energy_real.csv"
    
    if os.path.exists(data_path):
        print("✅ Real dataset found")
        return pd.read_csv(data_path)
    
    print("📥 Downloading REAL dataset from your Google Drive...")
    
    os.makedirs("data", exist_ok=True)
    
    try:
        # Your actual Google Drive link
        url = "https://drive.google.com/uc?export=download&id=1G--KX6I6WA4iiSejEVaqGi0EaMxspj2s"
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Download with progress
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(data_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"Progress: {percent:.1f}%", end='\r')
        
        print("\n✅ Real dataset downloaded successfully!")
        
        # Verify it's not empty
        file_size = os.path.getsize(data_path)
        if file_size > 1000:  # At least 1KB
            df = pd.read_csv(data_path)
            print(f"📊 Real dataset: {df.shape}")
            return df
        else:
            print("❌ Downloaded file is too small")
            return None
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

if __name__ == "__main__":
    download_real_dataset()
