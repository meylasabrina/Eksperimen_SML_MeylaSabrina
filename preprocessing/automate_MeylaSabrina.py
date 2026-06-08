import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    print("Mengunduh/Memuat data...")
    return pd.read_csv(file_path)

def clean_and_preprocess(df):
    print("Melakukan preprocessing data...")
    
    # 1. Hapus kolom tidak terpakai/kosong + kolom 'Floor' agar tidak lambat!
    drop_cols = ['Dimensions', 'Plot Area', 'Index', 'Title', 'Description', 'Amount(in rupees)', 
                 'location', 'Society', 'Super Area', 'overlooking', 'facing', 'Floor']
    df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore', inplace=True)
    
    # 2. Bersihkan target
    df.dropna(subset=['Price (in rupees)'], inplace=True)
    
    # 3. Ekstrak angka dari teks
    for col in ['Carpet Area', 'Bathroom', 'Balcony']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.extract(r'(\d+)').astype(float)
            
    # 4. Imputasi Missing Values
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols.remove('Price (in rupees)')
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # KODE BARU YANG BERSIH DAN AMAN:
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
        
    # 5. Encoding dengan uint8 (Hemat RAM)
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype='uint8')
    
    # 6. Scaling & Optimasi numpy array (Super Cepat)
    X = df_encoded.drop(columns=['Price (in rupees)'])
    y = df_encoded['Price (in rupees)'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    
    # Gabungkan horizontal level numpy
    all_data = np.hstack((X_scaled, y.reshape(-1, 1)))
    all_columns = list(X.columns) + ['target_price']
    
    df_clean = pd.DataFrame(all_data, columns=all_columns)
    return df_clean

def save_data(df, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'clean_data.csv')
    df.to_csv(output_path, index=False)
    print(f"Data bersih sukses disimpan di: {output_path}")

if __name__ == "__main__":
    # Path disesuaikan dengan posisi eksekusi dari root folder repository
    raw_data_path = "houseprices_raw/data.csv"
    output_directory = "preprocessing/house_prices"
    
    raw_df = load_data(raw_data_path)
    clean_df = clean_and_preprocess(raw_df)
    save_data(clean_df, output_directory)