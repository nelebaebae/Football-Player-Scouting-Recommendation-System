# Football Player Scouting Recommendation System

Repository ini berisi implementasi proyek skripsi tentang sistem rekomendasi pemain sepak bola berbasis pola sentuhan pemain di lapangan. Sistem mempelajari representasi aktivitas spasial pemain dari data `touch events`, lalu menghasilkan rekomendasi pemain yang memiliki pola permainan paling mirip dengan pemain acuan.

Metode utama yang digunakan adalah **Self-Supervised Learning (SSL)** untuk membentuk embedding dari touch map pemain, kemudian **cosine similarity** digunakan untuk menghitung kedekatan antar pemain. Aplikasi akhir dibangun menggunakan **Streamlit** dan dilengkapi visualisasi heatmap agar hasil rekomendasi lebih mudah diinterpretasikan.

## Ringkasan Skripsi

Topik penelitian ini berfokus pada pembangunan sistem scouting berbasis data yang dapat membantu proses pencarian pemain serupa. Sistem tidak hanya menampilkan daftar pemain yang direkomendasikan, tetapi juga memperlihatkan pola sebaran sentuhan pemain pada lapangan.

Secara umum, alur penelitian meliputi:

1. Mengolah data event sentuhan pemain dari dataset 1 liga dan 5 liga.
2. Mengubah koordinat sentuhan `x,y` menjadi representasi spatial grid atau touch map.
3. Melatih encoder CNN menggunakan pendekatan self-supervised contrastive learning.
4. Membentuk embedding pemain dari hasil encoder.
5. Menghitung similarity antar pemain menggunakan cosine similarity.
6. Mengevaluasi hasil terhadap baseline raw grid dan random baseline.
7. Menyajikan hasil rekomendasi dalam aplikasi Streamlit.

## Tujuan Penelitian

- Membangun sistem rekomendasi pemain sepak bola berdasarkan kemiripan pola aktivitas spasial.
- Membandingkan representasi berbasis SSL embedding dengan baseline raw spatial grid.
- Menyediakan visualisasi heatmap untuk membantu interpretasi hasil rekomendasi.
- Menyusun artefak aplikasi yang dapat digunakan untuk demo sistem scouting.

## Metode

### 1. Data Touch Events

Data utama berupa event sentuhan pemain yang memiliki informasi koordinat lapangan. Dataset yang digunakan:

| Dataset | Events setelah filter | Sampel player-match | Jumlah pemain |
| --- | ---: | ---: | ---: |
| 1 liga | 443.312 | 8.065 | 306 |
| 5 liga | 2.185.446 | 41.118 | 1.698 |

Dataset 5 liga digunakan sebagai ruang kandidat utama pada aplikasi, sedangkan dataset 1 liga digunakan sebagai analisis pendukung.

### 2. Spatial Grid / Touch Map

Setiap sentuhan pemain dipetakan ke grid lapangan. Representasi ini menggambarkan area mana yang paling sering digunakan pemain ketika melakukan open-play touches.

### 3. Self-Supervised Learning

Model SSL digunakan untuk mempelajari embedding pemain tanpa label similarity eksplisit. Pendekatan ini bertujuan agar model dapat menangkap pola spasial pemain secara lebih bermakna dibandingkan raw grid biasa.

### 4. Rekomendasi Pemain

Setelah embedding pemain terbentuk, sistem menghitung cosine similarity antara pemain acuan dan pemain lain. Pemain dengan nilai similarity tertinggi ditampilkan sebagai rekomendasi.

## Hasil Evaluasi

### Evaluasi Representasi

Evaluasi representasi membandingkan jarak kemiripan intra-player dan inter-player. Gap yang lebih besar menunjukkan representasi lebih mampu menjaga kemiripan sampel milik pemain yang sama sekaligus membedakannya dari pemain lain.

| Dataset | Representasi | Mean intra similarity | Mean inter similarity | Gap |
| --- | --- | ---: | ---: | ---: |
| 1 liga | Raw grid baseline | 0,3379 | 0,1615 | 0,1764 |
| 1 liga | SSL embedding | 0,3370 | 0,0030 | 0,3339 |
| 5 liga | Raw grid baseline | 0,3257 | 0,1622 | 0,1635 |
| 5 liga | SSL embedding | 0,3108 | 0,0030 | 0,3078 |

SSL embedding meningkatkan gap representasi sebesar:

| Dataset | Gap raw grid | Gap SSL | Peningkatan |
| --- | ---: | ---: | ---: |
| 1 liga | 0,1764 | 0,3339 | +0,1575 |
| 5 liga | 0,1635 | 0,3078 | +0,1443 |

### Evaluasi Proksi Rekomendasi

Evaluasi rekomendasi menggunakan konsistensi posisi pada top-5 rekomendasi.

| Dataset | Representasi | Kategori | Consistency@5 |
| --- | --- | --- | ---: |
| 1 liga | Random baseline | Role name | 0,3593 |
| 1 liga | Raw grid baseline | Role name | 0,8830 |
| 1 liga | SSL embedding | Role name | 0,8810 |
| 1 liga | Random baseline | Specific position | 0,1294 |
| 1 liga | Raw grid baseline | Specific position | 0,7085 |
| 1 liga | SSL embedding | Specific position | 0,7510 |
| 5 liga | Random baseline | Role name | 0,3514 |
| 5 liga | Raw grid baseline | Role name | 0,8605 |
| 5 liga | SSL embedding | Role name | 0,8575 |
| 5 liga | Random baseline | Specific position | 0,1306 |
| 5 liga | Raw grid baseline | Specific position | 0,7212 |
| 5 liga | SSL embedding | Specific position | 0,7188 |

Hasil ini menunjukkan bahwa rekomendasi berbasis pola sentuhan jauh lebih konsisten dibanding random baseline. SSL embedding menjadi metode utama aplikasi karena membentuk representasi yang lebih terpisah antara pemain berbeda, terutama pada evaluasi intra/inter similarity.

## Fitur Aplikasi

- Pilih dataset `1liga` atau `5liga`.
- Cari dan pilih anchor player.
- Tampilkan daftar rekomendasi pemain berdasarkan similarity.
- Tampilkan similarity dalam bentuk persentase relatif.
- Bandingkan heatmap anchor player dengan kandidat teratas.
- Tampilkan heatmap 4 kandidat teratas.
- Tampilkan tabel detail rekomendasi, termasuk posisi, tim, jumlah match, total touches, dan similarity score.

## Struktur Folder

```text
Football-Player-Scouting-Recommendation-System/
├── app.py
├── README.md
├── DATASET/
│   ├── LWRW.xlsx
│   ├── touch_events_1liga.csv
│   └── touch_events_5liga.csv
├── NOTEBOOK/
│   └── skripsi_final_SSL_vs_RawGrid_position_proxy.ipynb
├── artifacts/
│   └── player_recommender_artifacts.pkl
└── outputs_ssl_vs_rawgrid_position_proxy/
    ├── dataset_summary.csv
    ├── representation_evaluation_summary_raw_vs_ssl.csv
    ├── recommendation_proxy_evaluation_summary.csv
    ├── player_ssl_embeddings_1liga.csv
    ├── player_ssl_embeddings_5liga.csv
    ├── player_raw_grid_1liga.csv
    ├── player_raw_grid_5liga.csv
    └── file output evaluasi dan visualisasi lainnya
```

## Cara Menjalankan Aplikasi

Pastikan Python sudah terpasang, lalu jalankan perintah berikut dari folder repository:

```bash
cd Football-Player-Scouting-Recommendation-System
pip install streamlit pandas numpy matplotlib scipy
streamlit run app.py
```

Jika menggunakan virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install streamlit pandas numpy matplotlib scipy
streamlit run app.py
```

Aplikasi membutuhkan file artifact berikut:

```text
artifacts/player_recommender_artifacts.pkl
```

Jika artifact belum tersedia, jalankan notebook penelitian terlebih dahulu untuk membentuk embedding dan mengekspor artifact Streamlit.

## Notebook Penelitian

Notebook utama berada di:

```text
NOTEBOOK/skripsi_final_SSL_vs_RawGrid_position_proxy.ipynb
```

Notebook tersebut mencakup proses:

- loading dan preprocessing dataset,
- pembentukan touch map,
- training encoder SSL,
- pembentukan embedding pemain,
- evaluasi raw grid vs SSL embedding,
- evaluasi konsistensi rekomendasi,
- ekspor file output dan artifact aplikasi.

## Output Penting

Beberapa output penting untuk kebutuhan analisis skripsi:

- `dataset_summary.csv`: ringkasan jumlah data.
- `representation_evaluation_summary_raw_vs_ssl.csv`: evaluasi intra/inter similarity.
- `gap_improvement_ssl_vs_raw.csv`: peningkatan gap SSL dibanding raw grid.
- `recommendation_proxy_evaluation_summary.csv`: evaluasi Consistency@5.
- `position_consistency_raw_vs_ssl_vs_random.csv`: perbandingan random baseline, raw grid, dan SSL.
- `recommendation_ssl_5liga_m_ozil.csv`: contoh hasil rekomendasi SSL.
- `heatmap_comparison_5liga_m_ozil_top5.png`: contoh visualisasi heatmap rekomendasi.
- `player_recommender_artifacts.pkl`: artifact yang digunakan aplikasi Streamlit.

## Catatan

- Sistem ini menggunakan similarity berbasis pola sentuhan, bukan penilaian kualitas pemain secara menyeluruh.
- Hasil rekomendasi sebaiknya dipahami sebagai kandidat pemain dengan pola aktivitas spasial mirip.
- Faktor lain seperti usia, harga pasar, kondisi kontrak, performa teknis, dan konteks taktik tidak menjadi fokus utama pada model ini.
- Baseline raw grid digunakan sebagai pembanding penelitian, sedangkan aplikasi menampilkan hasil utama dari SSL embedding.

## Teknologi

- Python
- Pandas
- NumPy
- Matplotlib
- PyTorch
- Streamlit
- Self-Supervised Learning
- Cosine Similarity
