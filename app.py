import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern, hog
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load model
model = joblib.load('model_rf.pkl')

# Fungsi ekstraksi fitur
def extract_features_from_image(image):
    img = np.array(image)
    img = cv2.resize(img, (150, 150))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = edges.sum() / edges.size

    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)

    hog_feature = hog(gray, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)

    return np.concatenate([lbp_hist, hog_feature, [edge_density]])

# Styling background pink cerah tanpa navbar
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #fce4ec, #f3e5f5);
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Halaman tersedia
pages = ["Beranda", "Deteksi Gambar", "Tentang Model", "Nama Anggota Kelompok"]

# Sidebar Navigasi
clicked = st.sidebar.radio("Navigasi Menu:", pages)


# Halaman Beranda
if clicked == "Beranda":
    st.title("Deteksi Kerusakan Bangunan Berbasis Citra")

    st.title("Tes Gambar dari URL")

    st.markdown("""
    Aplikasi ini dikembangkan untuk membantu proses identifikasi kerusakan bangunan pascabencana menggunakan citra digital.

    ### Tujuan Aplikasi
    - Mempercepat proses inspeksi bangunan terdampak bencana
    - Mengurangi ketergantungan pada inspeksi manual
    - Memberikan alternatif evaluasi cepat berbasis AI

    ### Fitur Utama
    - Upload gambar bangunan
    - Deteksi otomatis: rusak atau tidak rusak
    - Hasil prediksi langsung ditampilkan

    ### Cara Menggunakan
    1. Masuk ke halaman Deteksi Gambar
    2. Upload gambar bangunan
    3. Tunggu beberapa detik, hasil akan muncul

    ### Cocok Digunakan Oleh:
    - Mahasiswa teknik sipil/informatika
    - Relawan kebencanaan
    - Dinas PU
    - Peneliti AI
    """)

# Halaman Deteksi Gambar
elif clicked == "Deteksi Gambar":
    st.title("Deteksi Kerusakan Bangunan")

    uploaded_file = st.file_uploader("Upload Gambar Bangunan", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang Diupload", use_container_width=True)

        with st.spinner("Mendeteksi..."):
            features = extract_features_from_image(image)
            prediction = model.predict([features])[0]

        if prediction == 1:
            st.success("Hasil Deteksi: Bangunan Rusak")
        else:
            st.success("Hasil Deteksi: Bangunan Tidak Rusak")

# Halaman Tentang Model
elif clicked == "Tentang Model":
    st.title("Tentang Model Deteksi")

    st.markdown("""
    Model yang digunakan adalah **Random Forest Classifier**, yaitu algoritma machine learning berbasis pohon keputusan yang bekerja secara ansambel.

    ### Fitur Ekstraksi:
    - **Edge Density**: Mengukur tepi/retakan
    - **LBP**: Mengidentifikasi tekstur permukaan
    - **HOG**: Menangkap pola arah dan kontur

    ### Evaluasi Model:
    - Akurasi: 75%
    - Presisi (kelas rusak): 100%
    - Recall (kelas rusak): 60%
    - F1-score: 75%

    Dataset berasal dari gambar bangunan terdampak gempa/bencana dan diproses untuk keseimbangan data. Model ini bisa dikembangkan dengan deep learning dan klasifikasi tingkat kerusakan.
    """)

# Halaman Nama Anggota Kelompok
elif clicked == "Nama Anggota Kelompok":
    st.title("Nama Anggota Kelompok")
    st.markdown("""
    - **Thania**  
    - **Anggi**  
    - **Nadya**  
    - **Uly**   
    - **Gita**
    """)
