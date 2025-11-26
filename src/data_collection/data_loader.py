import pandas as pd
import requests
import os

def download_real_dataset():
    """Download YOUR real dataset from Google Drive"""
    data_path = "data/europe_energy_real.csv"
    
    # Create data directory if not exists
    os.makedirs("data", exist_ok=True)
    
    if os.path.exists(data_path):
        print("✅ Dataset found locally")
        try:
            df = pd.read_csv(data_path)
            print(f"✅ Local dataset loaded: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Error reading local file: {e}")
            # Continue to download
    
    print("📥 Downloading dataset from Google Drive...")
    
    try:
        # YOUR Google Drive direct download link
        file_id = "1G--KX6I6WA4iiSejEVaqGi0EaMxspj2s"
        
        # Method 1: Direct download
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        print("🔄 Attempting download...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Check if we got a large file warning
        if "text/html" in response.headers.get('content-type', ''):
            print("⚠️  Large file detected, getting confirmation token...")
            
            # Extract confirmation token
            import re
            match = re.search(r'confirm=([^&]+)', response.text)
            if match:
                token = match.group(1)
                url = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
                response = requests.get(url, stream=True)
                response.raise_for_status()
        
        # Download the file
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(data_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"   Download progress: {percent:.1f}%", end='\r')
        
        print("\n✅ Download completed!")
        
        # Verify the file
        file_size = os.path.getsize(data_path)
        print(f"📦 File size: {file_size / 1024 / 1024:.2f} MB")
        
        if file_size == 0:
            print("❌ Downloaded file is empty!")
            os.remove(data_path)
            return None
        
        # Try to read the CSV
        try:
            df = pd.read_csv(data_path)
            print(f"✅ Dataset loaded successfully: {df.shape}")
            print(f"📊 Columns: {len(df.columns)}")
            print(f"📈 Sample columns: {list(df.columns)[:5]}...")
            return df
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        
        # Alternative method
        print("🔄 Trying alternative download method...")
        return try_alternative_download()

def try_alternative_download():
    """Alternative download method"""
    try:
        # Alternative URL format
        file_id = "1G--KX6I6WA4iiSejEVaqGi0EaMxspj2s"
        url = f"https://docs.google.com/uc?export=download&id={file_id}"
        
        response = requests.get(url, stream=True)
        
        with open("data/europe_energy_real.csv", 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print("✅ Alternative download completed!")
        return pd.read_csv("data/europe_energy_real.csv")
        
    except Exception as e:
        print(f"❌ Alternative download also failed: {e}")
        return None

def manual_download_instructions():
    """Show manual download instructions"""
    print("\n📋 MANUAL DOWNLOAD INSTRUCTIONS:")
    print("1. Open this link in your browser:")
    print("   https://drive.google.com/file/d/1G--KX6I6WA4iiSejEVaqGi0EaMxspj2s/view")
    print("2. Click the 'Download' button")
    print("3. Save the file as 'data/europe_energy_real.csv'")
    print("4. Run the script again")
    print("\n💡 Make sure the file is in the 'data' folder")
