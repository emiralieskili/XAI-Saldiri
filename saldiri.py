import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_prediction(premise, hypothesis, model, tokenizer):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    return probs

def main():
    model_path = "./modeller/hukuk_bert_model"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    class_names = ["contradiction", "entailment", "neutral"]

    # 1. Orijinal Metinler
    p_orig = "Davacı, elektrik faturası borcunu ödemeyen davalıya karşı icra takibi başlatmıştır."
    h_orig = "Davalı tarafın kredi kartı borcu yüzünden maaşına haciz konulmuştur."

    # 2. LIME Çıktılarına Göre Manipüle Edilmiş (Adversarial) Metinler
    # 'borcunu' -> 'ödemesini', 'icra takibi' -> 'hukuki süreç', 'haciz konulmuştur' -> 'bloke konulmuştur'
    p_adv = "Davacı, elektrik faturası ödemesini yapmayan davalıya karşı hukuki süreç başlatmıştır."
    h_adv = "Davalı tarafın kredi kartı borcu yüzünden maaşına bloke konulmuştur."

    # Tahminleri Al
    probs_orig = get_prediction(p_orig, h_orig, model, tokenizer)
    probs_adv = get_prediction(p_adv, h_adv, model, tokenizer)

    print("=== SALDIRI ÖNCESİ (ORİJİNAL) ===")
    print(f"Premise: {p_orig}")
    print(f"Hypothesis: {h_orig}")
    for idx, name in enumerate(class_names):
        print(f"{name}: %{probs_orig[idx]*100:.2f}")

    print("\n=== SALDIRI SONRASI (ADVERSARIAL) ===")
    print(f"Premise: {p_adv}")
    print(f"Hypothesis: {h_adv}")
    for idx, name in enumerate(class_names):
        print(f"{name}: %{probs_adv[idx]*100:.2f}")

if __name__ == "__main__":
    main()