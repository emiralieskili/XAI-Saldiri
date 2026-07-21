import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer
import numpy as np

def main():
    print("1. Donanım Kontrol Ediliyor ve Temizleniyor")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")

    print("\n2. Eğitilmiş model yükleniyor...")
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

    # LIME tek bir metin girdisi kabul ettiği için iki metin özel bir ayırıcıyla (" | ") birleştirilir.
    birlesik_metin = f"{premise} | {hypothesis}"

    # LIME'ın arka planda kelimeleri maskelerken kullanacağı BATCH OPTİMİZE EDİLMİŞ tahmin fonksiyonu.
    def predictor(texts):
        p_list = []
        h_list = []
        
        # Metinleri böl ve listelere topla
        for text in texts:
            parts = text.split(" | ")
            p_list.append(parts[0])
            h_list.append(parts[1] if len(parts) > 1 else "")

        all_probs = []
        batch_size = 32 # GPU belleğinin taşmaması için otuz ikişerlik paketler halinde işle

        for i in range(0, len(texts), batch_size):
            # İlgili paketlere ait premise (p) ve hypothesis (h) dilimlerini ayrıştır
            batch_p = p_list[i:i + batch_size]
            batch_h = h_list[i:i + batch_size]

            # Metinler GPU için tokenize et
            # Paket içindeki metinler BERT'in işleyebileceği PyTorch tensörlerine dönüştür
            # max_length=256 ve padding ile tüm girdiler sabit boyuta getirilerek matris oluştur
            inputs = tokenizer(
                batch_p, 
                batch_h, 
                return_tensors="pt", 
                truncation=True, 
                padding="max_length", 
                max_length=256
            )
            
            # Tensörleri GPU'ya taşı
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # GPU üzerinde toplu tahmin (Batch Inference)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        
            # Bellekte biriken geçici verileri temizle
            del inputs, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return np.vstack(all_probs)

    print("\n3. LIME Açıklayıcı hazırlanıyor...")
    explainer = LimeTextExplainer(
        class_names=class_names,
        split_expression=lambda text: [x for x in text.split() if x != "|"],
        random_state=42  # Sonuçların sabit ve tekrarlanabilir olması için
    )

    print("\n4. Karar açıklanıyor (GPU Hızlandırmalı)...")
    exp = explainer.explain_instance(
        birlesik_metin, 
        predictor, 
        num_features=10,
        num_samples=3000  # GPU aktif olduğu için 3000 yapmak çok zaman almaz
    )

    print("\n--- MODELİN KARARI VE GEREKÇESİ ---")
    print(f"Metin: {birlesik_metin}\n")
    
    tahmin_olasiliklari = predictor([birlesik_metin])[0]
    en_yuksek_sinif = np.argmax(tahmin_olasiliklari)
    print(f"Tahmin: {class_names[en_yuksek_sinif]} (%{tahmin_olasiliklari[en_yuksek_sinif]*100:.2f} güven oranı)")
    
    print("\nKararı en çok etkileyen kelimeler ve ağırlıkları:")
    for kelime, agirlik in exp.as_list():
        yon = "DESTEKLİYOR (+)" if agirlik > 0 else "ENGELLİYOR (-)"
        print(f"Kelime: '{kelime}' -> Ağırlık: {agirlik:.4f} ({yon})")

    exp.save_to_file("lime_aciklama.html")
    print("\nDetaylı görsel rapor 'lime_aciklama.html' olarak kaydedildi!")

if __name__ == "__main__":
    main()