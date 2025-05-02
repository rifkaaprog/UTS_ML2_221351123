import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load scaler dan label encoder
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')  # Bisa dihapus jika tidak digunakan

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="tour_travel_customer.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Prediksi Churn Customer Travel")
st.write("Masukkan informasi pelanggan untuk memprediksi apakah akan churn atau tidak.")

# Form input pengguna
age = st.slider("Umur", min_value=18, max_value=100, value=30)
frequent_flyer = st.selectbox("Frequent Flyer", ["Tidak", "Ya"])
income_class = st.selectbox("Kelas Penghasilan Tahunan", ["Rendah", "Menengah", "Tinggi"])  # Disesuaikan encode
services_opted = st.slider("Jumlah Layanan yang Digunakan", min_value=0, max_value=10, value=2)
social_media_sync = st.selectbox("Akun Tersambung Media Sosial", ["Tidak", "Ya"])
booked_hotel = st.selectbox("Pernah Memesan Hotel?", ["Tidak", "Ya"])

# Mapping ke angka (pastikan sama dengan preprocessing Anda saat training)
frequent_flyer_map = {"Tidak": 0, "Ya": 1}
income_class_map = {"Rendah": 0, "Menengah": 1, "Tinggi": 2}
social_media_map = {"Tidak": 0, "Ya": 1}
booked_hotel_map = {"Tidak": 0, "Ya": 1}

# Ketika tombol ditekan
if st.button("Prediksi"):
    input_data = np.array([[
        age,
        frequent_flyer_map[frequent_flyer],
        income_class_map[income_class],
        services_opted,
        social_media_map[social_media_sync],
        booked_hotel_map[booked_hotel]
    ]])

    input_scaled = scaler.transform(input_data).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = (prediction > 0.5).astype(int)[0][0]  # Asumsikan output biner (0 atau 1)

    if predicted_class == 1:
        st.warning("⚠️ Pelanggan berpotensi Churn (Berhenti menggunakan layanan).")
    else:
        st.success("✅ Pelanggan diprediksi tetap loyal.")