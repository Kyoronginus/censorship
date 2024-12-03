


import tweepy
import pandas as pd
import re

# Masukkan kredensial API Twitter Anda
API_KEY = '65tGUmHb9SaGo9UX0swBig9tO'
API_SECRET_KEY = 'ZL4VcdSfMtviw3uoJgqzhTimGoN5VpwhrSXzvuJazkbbNo9wN6'
ACCESS_TOKEN = '966300595903651840-eET2ZBFvgSXGhfmIEJOVBfUpEdnM7pb'
ACCESS_TOKEN_SECRET = 'QnaYgegZ0NhI8zJzGMXYC5MiuinvhxniZXKterUveIssa'

# Autentikasi dengan API Twitter
auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET_KEY, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

# Fungsi untuk mengumpulkan tweet
def collect_tweets(keyword, max_tweets=100):
    tweets = []
    for tweet in tweepy.Cursor(api.search_tweets, q=keyword, lang="id", tweet_mode='extended').items(max_tweets):
        tweets.append(tweet.full_text)
    return tweets

# Fungsi untuk menanyakan label ke user
def ask_for_labels(sentences):
    labeled_sentences = []
    print("\n=== Labeling Process ===")
    for sentence in sentences:
        print(f"\nKalimat: {sentence}")
        label = input("Masukkan label untuk kalimat ini (contoh: seksual, rasis, keterbatasan fisik/mental, lain): ").strip().lower()
        labeled_sentences.append([sentence, label])
    return labeled_sentences

# Fungsi untuk menyimpan dataset ke file CSV dan menambahkan label
def save_to_csv(sentences_with_labels, filename='dataset_twitter_anjing_labeled.csv'):
    # Buat DataFrame dengan kolom 'text' dan 'label'
    new_data = pd.DataFrame(sentences_with_labels, columns=['text', 'label'])

    try:
        # Cek apakah file CSV sudah ada, jika ya, tambahkan datanya
        existing_data = pd.read_csv(filename)
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
    except FileNotFoundError:
        # Jika file tidak ada, buat file baru
        updated_data = new_data
    
    # Simpan dataset ke CSV
    updated_data.to_csv(filename, index=False)
    print(f"Dataset updated and saved to {filename}")

# Fungsi utama untuk menjalankan scraping dan labeling
def main():
    # Input dari user untuk kata kunci
    keyword = input("Masukkan kata kunci untuk scraping (contoh: anjing): ").strip().lower()

    # Input dari user untuk jumlah tweet
    try:
        max_tweets = int(input("Masukkan jumlah tweet yang ingin diambil: ").strip())
    except ValueError:
        print("Input tidak valid. Menggunakan jumlah default 100.")
        max_tweets = 100

    # Mengumpulkan tweet yang mengandung kata kunci
    tweets = collect_tweets(keyword, max_tweets=max_tweets)

    # Jika tweet ditemukan, mulai proses labeling
    if tweets:
        sentences_with_labels = ask_for_labels(tweets)
        save_to_csv(sentences_with_labels)
    else:
        print(f"Tidak ditemukan tweet yang mengandung kata kunci '{keyword}'.")

# Menjalankan program utama
if __name__ == "__main__":
    main()
