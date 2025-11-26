import pandas as pd
import requests
import os

def download_real_dataset():
    """Download YOUR real dataset from Google Drive with proper handling"""
    data_path = "data/europe_energy_real.csv"
    
    # Create data directory if not exists
    os.makedirs("data", exist_ok=True)
    
    if os.path.exists(data_path):
        print("✅ Dataset found locally - checking if valid...")
        try:
            df = pd.read_csv(data_path)
            if len(df) > 0 and len(df.columns) > 1:
                print(f"✅ Valid local dataset: {df.shape}")
                return df
            else:
                print("❌ Local file is empty or invalid, re-downloading...")
                os.remove(data_path)
        except:
            print("❌ Local file corrupted, re-downloading...")
            os.remove(data_path)
    
    print("📥 Downloading dataset from Google Drive...")
    
    try:
        # Use the direct download link with confirmation
        file_id = "1G--KX6I6WA4iiSejEVaqGi0EaMxspj2s"
        
        # This is the correct URL format for large files
        url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
        
        print("🔄 Starting download (this may take a while for 124MB file)...")
        response = requests.get(url, stream=True, timeout=30)
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
                        print(f"   Download progress: {percent:.1f}%", end='\r')
        
        print("\n✅ Download completed!")
        
        # Verify the file
        file_size = os.path.getsize(data_path)
        print(f"📦 File size: {file_size / 1024 / 1024:.2f} MB")
        
        if file_size < 1000:  # If file is too small, it's probably HTML
            print("❌ File too small - likely got HTML page instead of CSV")
            os.remove(data_path)
            return None
        
        # Try to read the CSV
        try:
            print("🔍 Reading CSV file...")
            df = pd.read_csv(data_path)
            
            if len(df) == 0 or len(df.columns) < 2:
                print("❌ CSV file is empty or has too few columns")
                return None
                
            print(f"✅ Dataset loaded successfully: {df.shape}")
            print(f"📊 Columns: {len(df.columns)}")
            print(f"📈 First 5 columns: {list(df.columns)[:5]}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error reading CSV: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None

def manual_download_instructions():
    """Show manual download instructions"""
    print("\n" + "="*60)
    print("📋 MANUAL DOWNLOAD INSTRUCTIONS:")
    print("="*60)
    print("1. Open this link in your browser:")
    print("   https://drive.google.com/file/d/1G--KX6I6WA4iiSejEVaqGi0EaMxspj2s/view")
    print("\n2. You'll see a warning message:")
    print("   'Google Drive can't scan this file for viruses'")
    print("   'time_series_60min_singleindex.csv (124M) is too large'")
    print("\n3. Click the 'Download anyway' button")
    print("\n4. Save the file as 'europe_energy_real.csv' in the 'data' folder")
    print("\n5. Your folder structure should look like:")
    print("   your_project/")
    print("   ├── data/")
    print("   │   └── europe_energy_real.csv  ← This file")
    print("   ├── src/")
    print("   └── main.py")
    print("\n6. Run: python main.py")
    print("="*60)

def check_existing_file():
    """Check if manual download worked"""
    data_path = "data/europe_energy_real.csv"
    
    if os.path.exists(data_path):
        file_size = os.path.getsize(data_path)
        print(f"📦 File exists: {file_size / 1024 / 1024:.2f} MB")
        
        if file_size > 100000000:  # ~100MB
            print("✅ File size looks good (real dataset)")
            try:
                df = pd.read_csv(data_path, nrows=5)  # Read only first 5 rows to check
                print(f"✅ CSV is valid: {df.shape}")
                print(f"📊 Columns: {len(df.columns)}")
                print(f"🔍 First 3 columns: {list(df.columns)[:3]}")
                return df
            except Exception as e:
                print(f"❌ Error reading file: {e}")
                return None
        else:
            print("❌ File too small - might be corrupted")
            return None
    else:
        print("❌ File not found in data/ folder")
        return None
