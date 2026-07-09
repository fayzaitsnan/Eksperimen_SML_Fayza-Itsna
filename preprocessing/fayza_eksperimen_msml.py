import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File data mentah tidak ditemukan di: {file_path}")
    print(f"[1/4] Membaca data dari: {file_path}")
    return pd.read_csv(file_path)

def clean_duplicates(df):
    duplicate_count = df.duplicated().sum()
    print(f"[2/4] Ditemukan {duplicate_count} data duplikat.")
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print("      -> Data duplikat berhasil dihapus.")
    return df

def encode_features(df):
    """Mengubah seluruh fitur kategorikal menjadi numerik menggunakan LabelEncoder."""
    print("[3/4] Melakukan encoding fitur kategorikal...")
    le = LabelEncoder()
    
    columns_to_encode = df.columns
    
    for col in columns_to_encode:
        df[col] = le.fit_transform(df[col])
        
    print("      -> Seluruh kolom berhasil di-encode menjadi numerik.")
    return df

def save_clean_data(df, output_path):
    """Menyimpan hasil preprocessing ke file baru."""
    # Membuat folder tujuan jika belum ada
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[4/4] Data bersih berhasil disimpan di: {output_path}")

def main():
    input_file = '../survey lung cancer.csv'
    output_file = './survey_lung_cancer_clean.csv'
    
    print("=== Memulai Alur Otomatisasi Preprocessing Data ===")
    
    raw_df = load_data(input_file)
    dedup_df = clean_duplicates(raw_df)
    encoded_df = encode_features(dedup_df)
    save_clean_data(encoded_df, output_file)
    
    print("=== Proses Preprocessing Selesai dan Sukses! ===")

if __name__ == "__main__":
    main()