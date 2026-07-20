import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Modelin verilen iki metin arasındaki ilişkiyi tahmin etmesini sağlayan yardımcı fonksiyon
def get_prediction(premise, hypothesis, model, tokenizer):
    # İddia ve varsayım metinleri modelin işleyebileceği sayısal matris yapılarına dönüştürülür
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    # Gradyan takibi kapatılarak sadece ileri besleme (inference) hesaplaması yapılır
    with torch.no_grad():
        logits = model(**inputs).logits
        # Modelin ham skor çıktıları (logits) sınıf olasılıklarına dönüştürülür
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return probs

def main():
    # Kaydedilen modelin yolu
    model_path = "./modeller/hukuk_bert_model"
    # Kaydedilmiş model ağırlıklarını yükle
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()    # Modeli test moduna al

    class_names = ["contradiction", "entailment", "neutral"]    # Veri setindeki sınıflandırma etiketleri

    # 1. Orijinal Metinler
    p_orig = "Davacı, elektrik faturası borcunu ödemeyen davalıya karşı icra takibi başlatmıştır."
    h_orig = "Davalı tarafın kredi kartı borcu yüzünden maaşına haciz konulmuştur."

    # 2. LIME Çıktılarına Göre Manipüle Edilmiş (Adversarial) Metinler
    # İnsanlar için anlam korunurken, model için kritik olan "borcunu", "icra takibi" ve "haciz konulmuştur" ifadeleri
    # sırasıyla "ödemesini", "hukuki süreç" ve "bloke konulmuştur" kelimeleriyle değiştirilerek saldırı tasarlanmıştır
    p_adv = "Davacı, elektrik faturası ödemesini yapmayan davalıya karşı hukuki süreç başlatmıştır."
    h_adv = "Davalı tarafın kredi kartı borcu yüzünden maaşına bloke konulmuştur."

    # Hem orijinal hem de saldırı amaçlı hazırlanan metin çiftlerinin tahmin olasılıkları hesapla
    probs_orig = get_prediction(p_orig, h_orig, model, tokenizer)
    probs_adv = get_prediction(p_adv, h_adv, model, tokenizer)

    # Saldırı öncesi modelin ham metne verdiği yanıtlar ve sınıfların yüzde oranları yazdırılır
    print("=== SALDIRI ÖNCESİ (ORİJİNAL) ===")
    print(f"Premise: {p_orig}")
    print(f"Hypothesis: {h_orig}")
    for idx, name in enumerate(class_names):
        print(f"{name}: %{probs_orig[idx]*100:.2f}")

    # Saldırı sonrası modelin manipüle edilmiş metne verdiği yanıtlar yazdırılarak aradaki kararlılık farkı ölçülür
    print("\n=== SALDIRI SONRASI (ADVERSARIAL) ===")
    print(f"Premise: {p_adv}")
    print(f"Hypothesis: {h_adv}")
    for idx, name in enumerate(class_names):
        print(f"{name}: %{probs_adv[idx]*100:.2f}")

if __name__ == "__main__":
    main()