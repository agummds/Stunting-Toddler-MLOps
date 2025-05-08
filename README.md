# Eksperimen_SML_Agum Medisa

Repository ini berisi eksperimen machine learning untuk deteksi stunting pada balita.

## Struktur Repository

```
Eksperimen_SML_Agum Medisa
├── .workflow
│   └── preprocessing.yml
├── balita_stunting_status
│   └── data_balita.csv
├── preprocessing
│   ├── Eksperiment_Agum_Medisa.ipynb
│   ├── automate_agum-medisa.py
│   ├── requirements.txt
│   └── data_balita_preprocessing
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       └── preprocessors.joblib
└── README.md
```

## Workflow Otomatisasi

Repository ini menggunakan GitHub Actions untuk melakukan preprocessing data secara otomatis. Workflow akan dijalankan dalam situasi berikut:

1. Setiap kali ada perubahan pada file di folder `balita_stunting_status` atau `preprocessing`
2. Secara manual melalui GitHub Actions interface

Workflow akan:
1. Menginstall semua dependencies yang diperlukan
2. Menjalankan script preprocessing
3. Menyimpan hasil preprocessing ke folder `data_balita_preprocessing`
4. Melakukan commit dan push jika ada perubahan pada data yang sudah diproses

## Cara Penggunaan

1. Clone repository ini
2. Install dependencies:
   ```bash
   pip install -r preprocessing/requirements.txt
   ```
3. Jalankan preprocessing secara manual:
   ```bash
   python preprocessing/automate_agum-medisa.py
   ```

## Dataset

Dataset yang digunakan adalah [Stunting Toddler (Balita) Detection](https://www.kaggle.com/datasets/rendiputra/stunting-balita-detection-121k-rows) yang berisi data tentang:
- Umur (bulan)
- Jenis Kelamin
- Tinggi Badan (cm)
- Status Gizi 