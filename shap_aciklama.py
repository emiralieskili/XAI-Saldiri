import torch
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

def main():
    print("1. Cihaz Kontrol Ediliyor ve Temizleniyor")
    # GPU varsa GPU'yu seç
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # GPU'yu temizle
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"Kullanılan Cihaz: {str(device).upper()}")

    print("\n2. Eğitilmiş model yükleniyor...")
    # Eğitilmiş modeli yükle
    model_path = "./modeller/hukuk_bert_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Modeli ekran kartına taşı ve test moduna al
    model.to(device)
    model.eval()

    # Sınıf etiketleri
    class_names = ["contradiction", "entailment", "neutral"]

    # Örnek test davası metinlerinin tanımlanması
    premise = "Davacı, elektrik faturası borcunu ödemeyen davalıya karşı icra takibi başlatmıştır."
    hypothesis = "Davalı tarafın kredi kartı borcu yüzünden maaşına haciz konulmuştur."

    # Metin çifti ayırıcıyla birleştirilir
    birlesik_metin = f"{premise} | {hypothesis}"

    # SHAP için batch optimize edilmiş tahmin fonksiyonu
    def predictor(texts):
        p_list = []
        h_list = []
        
        # Metinleri böl ve listelere topla
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
                truncation=True, 
                padding="max_length", 
                max_length=256
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        
            del inputs, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.vstack(all_probs)

    print("\n3. SHAP Açıklayıcı hazırlanıyor...")
    # Maskelemede '|' simgesini ayırıcı olarak tanımlayarak hesaplamaya dahil etmiyoruz
    explainer = shap.Explainer(
        predictor, 
        masker=shap.maskers.Text(tokenizer=r"\s+"), 
        output_names=class_names
    )

    print("\n4. Karar açıklanıyor (GPU Hızlandırmalı)...")
    shap_values = explainer([birlesik_metin])

    print("\n--- MODELİN KARARI VE GEREKÇESİ ---")
    print(f"Metin: {birlesik_metin}\n")
    
    tahmin_olasiliklari = predictor([birlesik_metin])[0]
    en_yuksek_sinif = int(np.argmax(tahmin_olasiliklari))
    print(f"Tahmin: {class_names[en_yuksek_sinif]} (%{tahmin_olasiliklari[en_yuksek_sinif]*100:.2f} güven oranı)")
    
    # SHAP çıktısından kelime katkı skorlarını çekme (Entailment sınıfı için)
    entailment_idx = class_names.index("entailment")
    tokens = shap_values.data[0]
    scores = shap_values.values[0][:, entailment_idx]
    
    # '|' karakterini çıkartıp skorları virgülden sonra 4 haneye yuvarlıyoruz
    word_scores = [
        (tok.strip(), round(float(scr), 4)) for tok, scr in sorted(zip(tokens, scores), key=lambda x: abs(x[1]), reverse=True) 
        if tok.strip() != "|"
    ]

    print(f"\nKararı en çok etkileyen kelimeler ve ağırlıkları ({class_names[entailment_idx].upper()}):")
    for kelime, agirlik in word_scores:
        yon = "DESTEKLİYOR (+)" if agirlik > 0 else "ENGELLİYOR (-)"
        print(f"Kelime: '{kelime}' -> Ağırlık: {agirlik:.4f} ({yon})")

    # Görselleştirmeyi LIME Text Explainer formatına uyarlama
    from lime.lime_text import LimeTextExplainer
    lime_explainer = LimeTextExplainer(
        class_names=class_names,
        split_expression=lambda text: [x for x in text.split() if x != "|"],
        random_state=42
    )

    # LIME açıklama nesnesini oluşturma
    exp = lime_explainer.explain_instance(
        birlesik_metin, 
        predictor, 
        num_features=len(word_scores),
        num_samples=100,
        labels=(entailment_idx,)
    )
    
    # LIME haritalama yapısını doğrudan exp nesnesi üzerinden kelime-skor eşlemesi yapıyoruz
    exp.as_list = lambda label=entailment_idx: word_scores

    # LIME iç haritasını SHAP verileriyle güncelleme
    word_to_score = dict(word_scores)
    indexed_string = exp.domain_mapper.indexed_string
    local_exp_list = []
    
    for i in range(indexed_string.num_words()):
        word = indexed_string.word(i)
        if word in word_to_score:
            local_exp_list.append((i, word_to_score[word]))

    exp.local_exp[entailment_idx] = sorted(local_exp_list, key=lambda x: abs(x[1]), reverse=True)

    # HTML çıktısını alma ve özelleştirme
    raw_html = exp.as_html(labels=(entailment_idx,))
    
    # JS tarafında sol çubuk grafik dâhil tüm '.toFixed(2)' ve 'd3.format(".2f")' çıktılarını 4 basamağa zorluyoruz
    custom_html = raw_html.replace('.toFixed(2)', '.toFixed(4)') \
                           .replace('format(".2f")', 'format(".4f")') \
                           .replace('rgba(95, 186, 125', 'rgba(255, 127, 14') \
                           .replace('#5fba7d', '#ff7f0e') \
                           .replace('rgba(31, 119, 180', 'rgba(31, 119, 180')

    with open("shap_aciklama.html", "w", encoding="utf-8") as f:
        f.write(custom_html)

    print("\nDetaylı görsel rapor 'shap_aciklama.html' olarak kaydedildi!")

if __name__ == "__main__":
    main()