import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np

def main():
    print("1. Eğitilmiş model yükleniyor...")
    model_path = "./modeller/hukuk_bert_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval() # Modeli test moduna alıyoruz

    # Sınıf etiketlerimiz
    class_names = ["contradiction", "entailment", "neutral"]

    # 2. Örnek bir test davası (Premise ve Hypothesis birleşik formatta)
    # LIME metin üzerinde çalışırken tek bir metin girdisi bekler.
    # Bu yüzden araya BERT'in ayırıcı tokenı olan [SEP]'i koyarak birleştiriyoruz.
    premise = "Davacı, elektrik faturası borcunu ödemeyen davalıya karşı icra takibi başlatmıştır."
    hypothesis = "Davalı tarafın kredi kartı borcu yüzünden maaşına haciz konulmuştur."
    
    birlesik_metin = f"{premise} [SEP] {hypothesis}"

    # LIME için model tahmin fonksiyonu
    def predictor(texts):
        outputs = []
        for text in texts:
            parts = text.split("[SEP]")
            p = parts[0]
            h = parts[1] if len(parts) > 1 else ""
            
            inputs = tokenizer(p, h, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                outputs.append(probs)
        return np.array(outputs)

    print("\n2. LIME Açıklayıcı hazırlanıyor...")
    explainer = LimeTextExplainer(class_names=class_names)

    print("\n3. Karar açıklanıyor (Bu işlem 1-2 dakika sürebilir)...")
    # LIME kelimeleri rastgele maskeleyerek modelin tepki sınırlarını çözecek
    exp = explainer.explain_instance(
        birlesik_metin, 
        predictor, 
        num_features=10, # En etkili 10 kelimeyi göster
        num_samples=100  # Doğruluk için test örnek sayısı
    )

    print("\n--- MODELİN KARARI VE GEREKÇESİ ---")
    print(f"Metin: {birlesik_metin}\n")
    
    # Modelin ham tahminini hesapla
    tahmin_olasiliklari = predictor([birlesik_metin])[0]
    en_yuksek_sinif = np.argmax(tahmin_olasiliklari)
    print(f"Tahmin: {class_names[en_yuksek_sinif]} (%{tahmin_olasiliklari[en_yuksek_sinif]*100:.2f} güven oranı)")
    
    print("\nKararı en çok etkileyen kelimeler ve ağırlıkları:")
    for kelime, agirlik in exp.as_list():
        yon = "DESTEKLİYOR (+)" if agirlik > 0 else "ENGELLİYOR (-)"
        print(f"Kelime: '{kelime}' -> Ağırlık: {agirlik:.4f} ({yon})")

    # Sonuçları görsel HTML raporu olarak kaydet
    exp.save_to_file("lime_aciklama.html")
    print("\nDetaylı görsel rapor 'lime_aciklama.html' olarak kaydedildi!")

if __name__ == "__main__":
    main()