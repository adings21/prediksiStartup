from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load tokenizer dan model GPT-Neo 125M
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

# Buat pipeline dengan sampling agar output lebih variatif
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.7,
    top_p=0.95
)

def generate_ai_summary(input_data, prediction, prob_success):
    label = "BERHASIL" if prediction == 1 else "GAGAL"

    prompt = f"""
Berikut adalah data sebuah startup:
- Total pendanaan: {input_data['funding_total_usd_cleaned'][0]:,.0f} USD
- Jumlah putaran pendanaan: {input_data['funding_rounds'][0]}
- Negara: {input_data['country_code'][0]}
- Kategori utama: {input_data['primary_category'][0]}
- Usia startup: {input_data['startup_age_years'][0]} tahun

Model memprediksi bahwa startup ini akan **{label}**, dengan tingkat kepercayaan {prob_success:.2f}%.

Fitur terpenting dan kontribusinya:
- Primary Category: 30.77%
- Funding Amount: 30.51%
- Startup Age: 17.36%
- Country: 11.85%
- Funding Rounds: 8.28%

Tulis ringkasan analisis singkat mengapa hasil prediksi seperti itu, berdasarkan data dan kontribusi fitur.

Ringkasan:
"""

    try:
        result = pipe(prompt)[0]['generated_text']
        # Ambil hanya bagian setelah "Ringkasan:"
        return result.split("Ringkasan:")[-1].strip()
    except Exception as e:
        return f"❗ Gagal membuat ringkasan AI: {e}"
