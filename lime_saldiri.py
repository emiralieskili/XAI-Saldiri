import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

# LIME'ın türettiği yapay metinleri GPU/CPU üzerinde paketler halinde işleyen tahmin fonksiyonu
def predictor(texts, model, tokenizer, device):
    all_probs = []
    batch_size = 32  # GPU VRAM belleğinin dolmasını (OOM) önlemek için toplu işleme boyutu
    
    # Metin listesini 32'şerli paketler halinde döngüye al
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        premises = []
        hypotheses = []
        
        # ' | ' ayıracı ile birleştirilmiş metinleri iddia (premise) ve varsayım (hypothesis) olarak ayır
        for text in batch_texts:
            parts = text.split(" | ")
            premises.append(parts[0])
            hypotheses.append(parts[1] if len(parts) > 1 else "")
            
        # Metin çiftlerini BERT modelinin işleyebileceği sayısal tensörlere dönüştür
        inputs = tokenizer(
            premises, 
            hypotheses, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=256
        ).to(device)
        
        # Gradyan takibini kapatıp model tahminlerini hesapla
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()  # Ham skorları sınıf olasılıklarına dönüştür
            all_probs.append(probs)
            
        # Her işlem paketinden sonra geçici tensörleri ve GPU belleğini temizle
        del inputs, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    return np.vstack(all_probs)

def main():
    # Donanım kontrolü (GPU varsa CUDA, yoksa CPU kullanılır)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Eğitilmiş model ve tokenizer ağırlıklarının yükleneceği dizin
    model_path = "./modeller/hukuk_bert_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    model.to(device)
    model.eval()  # Modeli değerlendirme (test) moduna al

    class_names = ["contradiction", "entailment", "neutral"]  # Sınıf etiketleri

    # 1. Orijinal ve Adversarial (Saldırı Amaçlı Manipüle Edilmiş) Metin Çiftleri
    p_orig = "Davacı, elektrik faturası borcunu ödemeyen davalıya karşı icra takibi başlatmıştır."
    h_orig = "Davalı tarafın kredi kartı borcu yüzünden maaşına haciz konulmuştur."
    text_orig = f"{p_orig} | {h_orig}"

    p_adv = "Davacı, elektrik faturası ödemesini yapmayan davalıya karşı hukuki süreç başlatmıştır."
    h_adv = "Davalı tarafın kredi kartı borcu yüzünden maaşına bloke konulmuştur."
    text_adv = f"{p_adv} | {h_adv}"

    # 2. Hızlı Tahmin Karşılaştırması (Terminal Çıktısı)
    probs_orig = predictor([text_orig], model, tokenizer, device)[0]
    probs_adv = predictor([text_adv], model, tokenizer, device)[0]

    print("=== SALDIRI ÖNCESİ (ORİJİNAL) ===")
    for idx, name in enumerate(class_names):
        print(f"{name}: %{probs_orig[idx]*100:.2f}")

    print("\n=== SALDIRI SONRASI (ADVERSARIAL) ===")
    for idx, name in enumerate(class_names):
        print(f"{name}: %{probs_adv[idx]*100:.2f}")

    # 3. LIME Açıklayıcı Kurulumu
    # '|' karakterini kelime olarak değerlendirmemek için özel split fonksiyonu ve sabit seed tanımlandı
    explainer = LimeTextExplainer(
        class_names=class_names,
        split_expression=lambda x: [w for w in x.split() if w != "|"],
        random_state=42
    )

    # 4. Sadece Saldırı Metni İçin LIME Analizinin Çalıştırılması ve HTML Olarak Kaydedilmesi
    print("\nSaldırı metninin LIME açıklaması oluşturuluyor...")
    exp_adv = explainer.explain_instance(
        text_adv, 
        lambda x: predictor(x, model, tokenizer, device), 
        num_features=10, 
        num_samples=3000
    )
    exp_adv.save_to_file('saldiri_aciklama.html')

    print("İşlem tamamlandı! 'saldiri_aciklama.html' başarıyla oluşturuldu.")

if __name__ == "__main__":
    main()