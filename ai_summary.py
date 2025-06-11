from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load tokenizer dan model GPT-Neo 1.3B (lebih besar dan lebih pintar dari 125M)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")

# Buat pipeline dengan sampling agar output lebih variatif dan tidak mengulang
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2  # Mengurangi pengulangan yang tidak perlu
)

def generate_ai_summary(input_data, prediction, prob_success):
    label = "BERHASIL" if prediction == 1 else "GAGAL"

    prompt = f"""
Anda adalah analis data startup yang sangat ahli. Berikut adalah data tentang sebuah startup yang baru saja dievaluasi oleh model prediksi AI.

Data startup:
- Total pendanaan: {input_data['funding_total_usd_cleaned'][0]:,.0f} USD
- Jumlah putaran pendanaan: {input_data['funding_rounds'][0]}
- Negara: {input_data['country_code'][0]}
- Kategori utama: {input_data['primary_category'][0]}
- Usia startup: {input_data['startup_age_years'][0]} tahun

Model AI memprediksi bahwa startup ini akan **{label}** (GAGAL / BERHASIL), dengan tingkat kepercayaan **{prob_success:.2f}%**.

Analisis dan Ringkasan:
Jelaskan alasan mengapa model membuat prediksi ini berdasarkan data yang diberikan. Fokuskan pada kontribusi setiap faktor (misalnya: kategori utama, jumlah pendanaan, usia startup, dll) dan bagaimana hal itu mempengaruhi hasil.

Jawab dengan cara yang jelas dan padat, memberi alasan mengapa model memilih hasil prediksi ini dan apa yang menjadi faktor utama.

Ringkasan:
"""

    try:
        result = pipe(prompt)[0]['generated_text']
        # Ambil hanya bagian setelah "Ringkasan:"
        return result.split("Ringkasan:")[-1].strip()
    except Exception as e:
        return f"❗ Gagal membuat ringkasan AI: {e}"
