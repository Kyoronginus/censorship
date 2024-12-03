import pandas as pd

# Contoh dataset manual dengan 50 kalimat termasuk revisi cabul
data = [
    # Kalimat aman (label 0)
    ("Hari ini cuaca sangat cerah", 0),
    ("Saya suka bermain sepak bola dengan teman-teman", 0),
    ("Buku ini sangat menarik untuk dibaca", 0),
    ("Kucing itu sangat lucu dan menggemaskan", 0),
    ("Aku akan pergi ke taman besok pagi", 0),
    ("Dia adalah teman yang sangat baik", 0),
    ("Kami berencana untuk menonton film bersama", 0),
    ("Pemandangan gunung ini sungguh menakjubkan", 0),
    ("Saya akan memasak makanan favorit saya malam ini", 0),
    ("Perpustakaan ini memiliki banyak buku bagus", 0),
    ("Anjing peliharaan kami sangat pintar", 0),
    ("Saya senang belajar hal baru setiap hari", 0),
    ("Kue yang dia buat sangat enak", 0),
    ("Matahari terbenam tadi malam sangat indah", 0),
    ("Dia selalu membantu orang lain dengan tulus", 0),
    ("Kami menghabiskan waktu di pantai selama liburan", 0),
    ("Saya suka mendengarkan musik klasik saat bekerja", 0),
    ("Tempat ini sangat nyaman untuk bersantai", 0),
    ("Dia memenangkan lomba dengan usaha keras", 0),
    ("Saya akan mengunjungi kakek dan nenek akhir pekan ini", 0),

    # Kalimat kasar (label 1)
    ("Kamu ini benar-benar menyebalkan", 1),
    ("Dasar bodoh, tidak pernah mendengarkan!", 1),
    ("Dia itu orang yang sangat pemalas", 1),
    ("Kamu nggak berguna sama sekali", 1),
    ("Apa sih yang kamu lakukan, idiot?", 1),
    ("Jangan ganggu aku, kamu cuma bikin masalah", 1),
    ("Orang itu benar-benar brengsek", 1),
    ("Dia selalu ngomong tanpa mikir", 1),
    ("Kamu nggak punya otak atau apa?", 1),
    ("Kamu hanya memikirkan diri sendiri, dasar egois", 1),
    ("Sikapmu benar-benar bikin orang marah", 1),
    ("Kamu pengecut, nggak berani ambil risiko", 1),
    ("Dia ngomong kasar setiap kali marah", 1),
    ("Berhenti sok tahu dan diamlah", 1),
    ("Kamu ini nggak pernah bisa diandalkan", 1),

    # Kalimat cabul (label 2) yang lebih ofensif
    ("Bilang saja kalau kamu mau tidur dengan dia, dasar mesum", 2),
    ("Foto bugil yang kamu kirim benar-benar menjijikkan", 2),
    ("Jangan kirim gambar cabul seperti itu di grup!", 2),
    ("Apa kamu memang suka nonton film porno sepanjang waktu?", 2),
    ("Orang cabul seperti kamu hanya bisa mikirin seks terus", 2),
    ("Bocah itu terlalu muda untuk lihat konten bokep", 2),
    ("Jangan kirim konten mesum di tempat kerja!", 2),
    ("Dia terang-terangan menyentuh orang lain tanpa izin, dasar cabul!", 2),
    ("Konten seksual yang kamu kirim benar-benar bikin mual", 2),
    ("Kata-kata mesummu itu tidak pantas didengar siapapun", 2),
    ("Si brengsek itu malah ngirimin video porno ke grup", 2),
    ("Pesan itu jelas-jelas cabul dan tidak sopan", 2),
    ("Dia cabul banget, nggak bisa berhenti ngomongin seks", 2),
    ("Sebaiknya kamu simpan fantasi kotormu untuk dirimu sendiri", 2),
    ("Orang itu cabul parah, dia terang-terangan ngomongin hal vulgar", 2),
]

# Membuat DataFrame
df = pd.DataFrame(data, columns=['text', 'label'])

# Simpan dataset ke file CSV
df.to_csv('dataset_50_cabul_revisi.csv', sep=';', index=False)
print("Dataset saved to 'dataset_50_cabul_revisi.csv'")
