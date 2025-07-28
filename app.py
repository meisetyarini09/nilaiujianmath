import streamlit as st
import pandas as pd
import joblib

# Load model (pastikan file 'model.pkl' ada di folder yang sama)
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# Judul Aplikasi
st.title("Prediksi Skor Ujian Berdasarkan Data Akademik")

st.subheader("Masukkan Data Siswa:")

# Input pengguna
new_Sleep_Hours = st.number_input("Sleep_Hours (Jam tidur per hari):", min_value=0.0, step=0.5)
new_Hours_Studied = st.number_input("Hours_Studied (Jam belajar per hari):", min_value=0.0, step=0.5)
new_Attendance = st.number_input("Attendance (Persentase kehadiran):", min_value=0.0, max_value=100.0, step=1.0)
new_Previous_Scores = st.number_input("Previous_Scores (Nilai sebelumnya):", min_value=0.0, max_value=100.0, step=1.0)

# Prediksi ketika tombol ditekan
if st.button("Prediksi Skor Ujian"):
    try:
        # Siapkan data dalam bentuk DataFrame
        new_data_df = pd.DataFrame(
            [[new_Sleep_Hours, new_Hours_Studied, new_Attendance, new_Previous_Scores]],
            columns=['Sleep_Hours', 'Hours_Studied', 'Attendance', 'Previous_Scores']
        )

        # Prediksi menggunakan model
        predicted_income = model.predict(new_data_df)

        # Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi:")
        st.write(f"Skor ujian yang diprediksi: **{predicted_income[0][0]:,.2f}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
