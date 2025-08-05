import pandas as pd
import os
import glob

def load_all_postback_data(base_path, max_partitions=None):
    """Load more postback data to find FTD events"""
    
    partition_path = os.path.join(
        base_path, 
        "data_winline_kz", 
        "processed", 
        "dataset_type=postback", 
        "domain=postbacks.metaratings.ru"
    )
    
    date_folders = glob.glob(os.path.join(partition_path, "event_date=*"))
    date_folders = sorted(date_folders)
    
    if max_partitions:
        date_folders = date_folders[:max_partitions]
    
    print(f"Loading {len(date_folders)} partitions...")
    
    all_data = []
    
    for i, date_folder in enumerate(date_folders):
        date_str = os.path.basename(date_folder).split("event_date=")[1]
        parquet_files = glob.glob(os.path.join(date_folder, "*.parquet"))
        
        for parquet_file in parquet_files:
            df_chunk = pd.read_parquet(parquet_file, engine='auto')
            df_chunk['event_date'] = date_str
            all_data.append(df_chunk)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} partitions...")
    
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        return df
    else:
        return pd.DataFrame()



def load_ga4_events_data(ga4_path, max_partitions=10):
    """Load GA4 events data - the 7.16M events with page views and clicks"""
    
    print(f"Loading GA4 data from: {ga4_path}")
    
    if not os.path.exists(ga4_path):
        print(f"❌ GA4 path not found: {ga4_path}")
        return pd.DataFrame()
    
    # Find all event_date partitions
    date_folders = glob.glob(os.path.join(ga4_path, "event_date=*"))
    
    print(f"Found {len(date_folders)} GA4 date partitions")
    
    if max_partitions:
        date_folders = sorted(date_folders)[:max_partitions]
        print(f"Loading first {max_partitions} partitions for testing...")
    
    all_data = []
    
    for i, date_folder in enumerate(date_folders):
        date_str = os.path.basename(date_folder).split("event_date=")[-1].replace("_", "")
        
        # Find parquet files in this partition
        parquet_files = glob.glob(os.path.join(date_folder, "*.parquet"))
        
        print(f"Processing {date_str}: {len(parquet_files)} parquet files")
        
        for parquet_file in parquet_files:
            try:
                df_chunk = pd.read_parquet(parquet_file, engine='auto')
                df_chunk['event_date'] = date_str
                all_data.append(df_chunk)
                
                print(f"  ✅ Loaded {len(df_chunk)} rows from {os.path.basename(parquet_file)}")
                
            except Exception as e:
                print(f"  ❌ Error loading {parquet_file}: {e}")
        
        if (i + 1) % 2 == 0:
            print(f"Processed {i + 1} partitions...")
    
    if all_data:
        df = pd.concat(all_data, ignore_index=True)
        print(f"\n Done. Combined GA4 dataset: {len(df):,} rows")
        return df
    else:
        print("❌ No GA4 data loaded")
        return pd.DataFrame()