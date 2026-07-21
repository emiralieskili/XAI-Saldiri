import torch
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from lime.lime_text import LimeTextExplainer

def main():
    print("1. Cihaz Kontrol Ediliyor ve Temizleniyor")
    # Cihaz GPU ise GPU'yu seç
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # GPU'yu temizle
    if device.type == "cuda":
        torch.cuda.empty_cache() 
    print(f"Kullanılan Cihaz: {str(device).upper()}")

    print("\n2. Eğitilmiş Model Yükleniyor")
    # Eğitilmiş modeli yükle
    model_path = "./modeller/hukuk_bert_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    class_names = ["contradiction", "entailment", "neutral"]    # Çelişkili, ilişkili, nötr sınıfları
    entailment_idx = class_names.index("entailment")

    # Adversarial (Saldırı) Metin Çifti
    p_adv = "Davacı, elektrik faturası ödemesini yapmayan davalıya karşı hukuki süreç başlatmıştır."
    h_adv = "Davalı tarafın kredi kartı borcu yüzünden maaşına bloke konulmuştur."

    # Metin çifti ayırıcıyla birleştirilir
    text_clean_adv = f"{p_adv} | {h_adv}"

    # SHAP için metin dizilerini model tahmin olasılıklarına dönüştüren fonksiyon
    def predict_pipeline(texts):
        p_list = []
        h_list = []
        
        for text in texts:
            parts = text.split(" | ")
            p_list.append(parts[0])
            h_list.append(parts[1] if len(parts) > 1 else "")

        all_probs = []
        batch_size = 32

        for i in range(0, len(texts), batch_size):
            batch_p = p_list[i:i + batch_size]
            batch_h = h_list[i:i + batch_size]

            inputs = tokenizer(
                batch_p, 
                batch_h, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=256
            ).to(device)
            
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        
            del inputs, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.vstack(all_probs)

    print("\n3. SHAP Açıklayıcı Hazırlanıyor")
    # SHAP Explainer Kurulumu (| simgesi maskelemeye dahil edilmiyor)
    explainer = shap.Explainer(
        predict_pipeline, 
        masker=shap.maskers.Text(tokenizer=r"\s+"), 
        output_names=class_names
    )

    print("\n4. Karar açıklanıyor (GPU Hızlandırmalı)...")
    shap_values = explainer([text_clean_adv])

    print("\n--- MODELİN KARARI VE GEREKÇESİ ---")
    print(f"Metin: {text_clean_adv}\n")

    tahmin_olasiliklari = predict_pipeline([text_clean_adv])[0]
    en_yuksek_sinif = int(np.argmax(tahmin_olasiliklari))
    print(f"Tahmin: {class_names[en_yuksek_sinif]} (%{tahmin_olasiliklari[en_yuksek_sinif]*100:.2f} güven oranı)")

    # SHAP çıktısından kelime katkı skorlarını çekme (Entailment sınıfı için)
    tokens = shap_values.data[0]
    scores = shap_values.values[0][:, entailment_idx]
    
    # '|' karakterini çıkartıyoruz
    word_scores = [
        (tok.strip(), float(scr)) for tok, scr in sorted(zip(tokens, scores), key=lambda x: abs(x[1]), reverse=True) 
        if tok.strip() != "|"
    ]

    print(f"\nKararı en çok etkileyen kelimeler ve ağırlıkları ({class_names[entailment_idx].upper()}):")
    for kelime, agirlik in word_scores:
        yon = "DESTEKLİYOR (+)" if agirlik > 0 else "ENGELLİYOR (-)"
        print(f"Kelime: '{kelime}' -> Ağırlık: {agirlik:.4f} ({yon})")

    # Görselleştirmeyi LIME Text Explainer formatına uyarlama
    lime_explainer = LimeTextExplainer(
        class_names=class_names,
        split_expression=lambda text: [x for x in text.split() if x != "|"],
        random_state=42
    )

    # LIME açıklama nesnesini oluşturma
    exp = lime_explainer.explain_instance(
        text_clean_adv, 
        predict_pipeline, 
        num_features=len(word_scores),
        num_samples=100,
        labels=(entailment_idx,)
    )
    
    # LIME haritalama yapısını SHAP verileriyle güncelleme
    exp.as_list = lambda label=entailment_idx: word_scores

    word_to_score = dict(word_scores)
    indexed_string = exp.domain_mapper.indexed_string
    local_exp_list = []
    
    for i in range(indexed_string.num_words()):
        word = indexed_string.word(i)
        if word in word_to_score:
            local_exp_list.append((i, word_to_score[word]))

    exp.local_exp[entailment_idx] = sorted(local_exp_list, key=lambda x: abs(x[1]), reverse=True)

    # HTML çıktısını alma ve Entailment (pozitif) için Turuncu rengi entegre etme
    raw_html = exp.as_html(labels=(entailment_idx,))
    custom_html = raw_html.replace('rgba(95, 186, 125', 'rgba(255, 127, 14') \
                           .replace('#5fba7d', '#ff7f0e') \
                           .replace('rgba(31, 119, 180', 'rgba(31, 119, 180')

    with open("shap_saldiri_aciklama.html", "w", encoding="utf-8") as f:
        f.write(custom_html)

    print("\nDetaylı görsel rapor 'shap_saldiri_aciklama.html' olarak kaydedildi!")

if __name__ == "__main__":
    main()