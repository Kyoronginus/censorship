import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

# Fungsi untuk mengumpulkan kalimat yang mengandung kata 'anjing'
def collect_sentences_with_anjing(url):
    try:
        # Mengambil konten halaman web
        response = requests.get(url)
        response.raise_for_status()  # Memeriksa apakah request berhasil
        
        # Parsing konten halaman dengan BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Mengambil semua paragraf dari halaman (dapat disesuaikan tergantung struktur situs)
        paragraphs = soup.find_all('p')
        
        # Daftar untuk menyimpan kalimat yang mengandung kata 'anjing'
        sentences = []
        
        # Memproses setiap paragraf
        for paragraph in paragraphs:
            text = paragraph.get_text()
            # Split paragraf menjadi kalimat
            for sentence in re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text):
                # Cari kalimat yang mengandung kata 'anjing'
                if 'anjing' in sentence.lower():
                    sentences.append(sentence.strip())
        
        return sentences
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return []

# Fungsi untuk menyimpan dataset ke file CSV
def save_to_csv(sentences, filename='dataset_anjing.csv'):
    df = pd.DataFrame(sentences, columns=['text'])
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")

# Daftar URL sumber data (bisa diperluas)
urls = [
    'https://id.wikipedia.org/wiki/Anjing'
]

# Mengumpulkan kalimat dari berbagai URL
all_sentences = []
for url in urls:
    sentences = collect_sentences_with_anjing(url)
    all_sentences.extend(sentences)

# Simpan dataset ke CSV jika kalimat ditemukan
if all_sentences:
    save_to_csv(all_sentences)
else:
    print("No sentences containing 'anjing' were found.")
