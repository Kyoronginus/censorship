import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

# Fungsi untuk mengumpulkan kalimat yang mengandung kata kunci yang dimasukkan user
def collect_sentences_with_keyword(url, keyword):
    try:
        # Mengambil konten halaman web
        response = requests.get(url)
        response.raise_for_status()  # Memeriksa apakah request berhasil
        
        # Parsing konten halaman dengan BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Mengambil semua paragraf dari halaman (dapat disesuaikan tergantung struktur situs)
        paragraphs = soup.find_all('p')
        
        # Daftar untuk menyimpan kalimat yang mengandung kata kunci
        sentences = []
        
        # Memproses setiap paragraf
        for paragraph in paragraphs:
            text = paragraph.get_text()
            # Split paragraf menjadi kalimat
            for sentence in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text):
                # Cari kalimat yang mengandung kata kunci
                if keyword in sentence.lower():
                    sentences.append(sentence.strip())
        
        return sentences
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []

# Fungsi untuk menyimpan dataset ke file CSV dan menambahkan label
def save_to_csv(sentences_with_labels, keyword):
    # Buat nama file berdasarkan keyword
    filename = f"{keyword}.csv"

    # Buat DataFrame dengan kolom 'text' dan 'label'
    new_data = pd.DataFrame(sentences_with_labels, columns=['text', 'label'])

    try:
        # Cek apakah file CSV sudah ada, jika ya, tambahkan datanya
        existing_data = pd.read_csv(filename, sep=';')  # Memastikan CSV lama diparsing dengan delimiter ;
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    except FileNotFoundError:
        # Jika file tidak ada, buat file baru
        updated_data = new_data
    
    # Simpan dataset ke CSV dengan delimiter ;
    updated_data.to_csv(filename, index=False, sep=';')  # Gunakan delimiter ;
    print(f"Dataset updated and saved to {filename}")



# Fungsi untuk menanyakan label ke user
def ask_for_labels(sentences):
    labeled_sentences = []
    print("\n=== Labeling Process ===")
    for sentence in sentences:
        print(f"\nKalimat: {sentence}")
        label = input("Masukkan label untuk kalimat ini (contoh: 0 untuk suitable, 1 untuk rough, 2 untuk sexual: ").strip().lower()
        labeled_sentences.append([sentence, label])
    return labeled_sentences

# Fungsi utama untuk menjalankan scraping dan labeling
def main():
    # Input dari user untuk kata kunci
    keyword = input("Masukkan kata kunci untuk scraping (contoh: anjing): ").strip().lower()

    # Input dari user untuk URL
    urls = []
    while True:
        url = input("Masukkan URL untuk scraping (ketik 'selesai' jika sudah selesai): ").strip()
        if url.lower() == 'selesai':
            break
        urls.append(url)

    if not urls:
        print("Tidak ada URL yang dimasukkan. Program dihentikan.")
        return

    # Mengumpulkan kalimat dari berbagai URL
    all_sentences = []
    for url in urls:
        sentences = collect_sentences_with_keyword(url, keyword)
        all_sentences.extend(sentences)

    # Jika kalimat ditemukan, mulai proses labeling
    if all_sentences:
        sentences_with_labels = ask_for_labels(all_sentences)
        save_to_csv(sentences_with_labels,keyword)
    else:
        print(f"Tidak ditemukan kalimat yang mengandung kata kunci '{keyword}'.")

# Menjalankan program utama
if __name__ == "__main__":
    main()
