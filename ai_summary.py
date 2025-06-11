from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load tokenizer dan model GPT-Neo 125M
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

# Buat pipeline dengan setting untuk hasil variatif dan tidak mengulang
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

def generate_ai_summary(input_data, prediction, prob_success):
    label = "BERHASIL" if prediction == 1 else "GAGAL"

    prompt = f"""
Anda adalah analis startup yang ahli dalam membaca data dan menjelaskan hasil prediksi dari model AI.

Berikut adalah informasi tentang sebuah startup:
- Total pendanaan: {input_data['funding_total_usd_cleaned'][0]:,.0f} USD
- Jumlah putaran pendanaan: {input_data['funding_rounds'][0]}
- Negara: {input_data['country_code'][0]}
- Kategori utama: {input_data['primary_category'][0]}
- Usia startup: {input_data['startup_age_years'][0]} tahun

Model AI memprediksi bahwa status startup ini adalah **{label}**, dengan tingkat keyakinan sebesar {prob_success:.2f}%.

Kontribusi fitur utama terhadap prediksi:
- Kategori Utama: 30.77%
- Total Pendanaan: 30.51%
- Usia Startup: 17.36%
- Negara: 11.85%
- Putaran Pendanaan: 8.28%

Buatlah ringkasan analisis berdasarkan data di atas. Jelaskan secara singkat dan padat alasan mengapa model memprediksi hasil tersebut.

Analisis:
"""

    try:
        result = pipe(prompt)[0]['generated_text']
        # Ambil hanya bagian setelah "Analisis:"
        return result.split("Analisis:")[-1].strip()
    except Exception as e:
        return f"❗ Gagal membuat ringkasan AI: {e}"
