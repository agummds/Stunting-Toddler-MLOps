Tentu, saya akan membantu Anda memperbaiki file README untuk repositori Anda. Berikut adalah versi yang lebih terstruktur dan komprehensif yang menggabungkan informasi dari semua file yang Anda berikan.

-----

# Proyek MLOps: Deteksi Stunting pada Balita

Repositori ini berisi proyek *machine learning* untuk mendeteksi stunting pada balita, yang diimplementasikan dengan alur kerja MLOps lengkap, mulai dari pemrosesan data, pelatihan model, hingga pemantauan (*monitoring*).

## 📝 Deskripsi Proyek

Proyek ini bertujuan untuk membangun model *Random Forest Classifier* yang dapat memprediksi status stunting pada balita berdasarkan data seperti umur, jenis kelamin, dan tinggi badan. Seluruh alur kerja, mulai dari eksperimen hingga *deployment*, diotomatisasi dan dipantau menggunakan berbagai *tools* MLOps.

  - **Repositori GitHub**: [https://github.com/agummds/Stunting-Toddler-MLOps](https://github.com/agummds/Stunting-Toddler-MLOps)
  - **Repositori DagsHub & MLflow Tracking**: [https://dagshub.com/agummds/Stunting-Toddler](https://dagshub.com/agummds/Stunting-Toddler)

## 📁 Struktur Repositori

```
Stunting-Toddler-MLOps/
├── .github/workflows/
│   ├── mlflow-ci.yml         # Workflow CI/CD untuk training & deployment model
│   └── preprocessing.yml     # Workflow untuk otomatisasi preprocessing data
├── Membangun_model/
│   ├── modelling.py          # Script utama untuk training model
│   ├── MLProject             # Konfigurasi proyek MLflow
│   ├── conda.yaml            # Dependensi environment untuk training
│   └── models/               # Output model yang disimpan
├── Monitoring dan Logging/
│   ├── 3.prometheus_exporter.py # Exporter metrik untuk Prometheus
│   ├── 7.Inference.py        # API Flask untuk inferensi model
│   ├── prometheus.yml        # Konfigurasi Prometheus
│   └── bukti monitoring Grafana/
│       └── dashboard.json    # Konfigurasi dashboard Grafana
└── README.md
```

-----

## ⚙️ Alur Kerja MLOps

Proyek ini mengimplementasikan beberapa komponen MLOps utama:

### 1\. Otomatisasi Preprocessing Data

  - Repositori ini menggunakan **GitHub Actions** untuk menjalankan *preprocessing* data secara otomatis setiap kali ada perubahan pada folder `data/` atau `preprocessing/`.
  - *Workflow* akan menginstal dependensi, menjalankan skrip *preprocessing*, dan menyimpan hasilnya (seperti `X_train.csv`, `X_test.csv`, dll.) ke direktori `preprocessing/data_balita_preprocessing/`.
  - Perubahan pada data yang telah diproses akan di-*commit* dan sebuah *pull request* akan dibuat secara otomatis ke *branch master*.

### 2\. Pelatihan Model dan Eksperimen

  - Proses pelatihan model diatur menggunakan **MLflow**. Skrip `modelling.py` akan melatih model *Random Forest* dan mencatat berbagai hal ke **DagsHub**.
  - **Eksperimen yang dilacak**:
      - **Parameter**: `n_estimators`, `max_depth`.
      - **Metrik**: Akurasi, presisi, recall, F1-score, dan metrik kustom seperti `prediction_confidence`.
      - **Artefak**: Model tersimpan, `confusion_matrix.png`, dan `feature_importance.png`.

### 3\. Continuous Integration & Deployment (CI/CD)

  - *Workflow* CI/CD diatur dalam file `.github/workflows/mlflow-ci.yml`.
  - **Pemicu**: *Workflow* berjalan setiap kali ada *push* atau *pull request* ke *branch* `main`.
  - **Tahapan *Workflow***:
    1.  **Train**: Menjalankan *MLflow project* untuk melatih model dan menyimpan artefak ke DagsHub.
    2.  **Docker**: Membangun *image* Docker yang berisi API inferensi dan model, lalu mendorongnya ke DockerHub.

### 4\. Pemantauan (Monitoring) & Logging

  - Model yang di-*deploy* diekspos melalui **Prometheus Exporter** untuk memantau performanya secara *real-time*.
  - **Metrik yang Dipantau**:
      - `model_predictions_total`: Jumlah total prediksi.
      - `model_prediction_latency_seconds`: Latensi prediksi.
      - `model_cpu_usage_percent`: Penggunaan CPU.
      - `model_memory_usage_bytes`: Penggunaan memori.
      - `model_errors_total`: Jumlah total *error*.
  - Metrik ini divisualisasikan menggunakan **Grafana** dengan *dashboard* yang telah dikonfigurasi (`dashboard.json`).
  - Sistem *alerting* juga diatur di Grafana untuk memberitahu jika terjadi anomali seperti penggunaan CPU yang tinggi atau tingkat *error* yang tinggi.

### 5\. Inferensi Model

  - Model yang telah dilatih disajikan melalui **API Flask** (`7.Inference.py`).
  - *Endpoint* `/predict` menerima data input dalam format JSON dan mengembalikan hasil prediksi.
  - *Endpoint* `/health` tersedia untuk memeriksa status kesehatan layanan.

-----

## 🚀 Cara Menjalankan Proyek

### Prasyarat

  - Conda
  - Docker

### 1\. Menjalankan Pelatihan Model (Lokal)

1.  **Buat dan aktifkan *environment* Conda:**
    ```bash
    conda env create -f Membangun_model/conda.yaml
    conda activate stunting-prediction
    ```
2.  **Atur Variabel Lingkungan:**
    Buat file `.env` di dalam folder `Membangun_model/` dan isi dengan kredensial DagsHub Anda.
    ```
    MLFLOW_TRACKING_USERNAME=username_dagshub
    MLFLOW_TRACKING_PASSWORD=token_dagshub
    ```
3.  **Jalankan *MLflow Project*:**
    ```bash
    mlflow run Membangun_model --experiment-name "stunting-prediction"
    ```

### 2\. Menjalankan dengan Docker

Proyek ini juga dapat dijalankan sebagai kontainer Docker setelah *image* berhasil di-*build* melalui *workflow* CI/CD.

```bash
docker run -p 5000:5000 -e MODEL_PATH='path/to/model' username_docker/stunting-prediction:latest
```

-----

## 📊 Dataset

Dataset yang digunakan adalah [Stunting Toddler (Balita) Detection](https://www.kaggle.com/datasets/rendiputra/stunting-balita-detection-121k-rows) yang berisi informasi mengenai:

  - Umur (dalam bulan)
  - Jenis Kelamin
  - Tinggi Badan (cm)
  - Status Gizi