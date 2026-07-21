import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np

def encode_labels(example):
    # Metinsel etiketleri sayısal değerlere dönüştür
    mapping = {"contradiction": 0, "entailment": 1, "neutral": 2}
    # Eğer etiket veri setinde beklenmeyen bir formatta ise varsayılan olarak 2 (neutral) ver
    example["label"] = mapping.get(str(example["label"]).strip().lower(), 2)
    return example

def main():
    print("1. Cihaz kontrol ediliyor...")
    device = "cuda" if torch.cuda.is_available() else "cpu" # GPU aktifse GPU'yu değilse CPU'u kullan
    print(f"Kullanılan Cihaz: {str(device).upper()}")

    print("\n2. Veri seti yükleniyor...")
    dataset = load_dataset("Turkish-NLI/legal_nli_TR_V1")
    
    # Tüm verilerle modeli eğit
    train_sub = dataset["train"].map(encode_labels)
    test_sub = dataset["test"].map(encode_labels)

    print(f"Eğitim verisi boyutu: {len(train_sub)} \nTest verisi boyutu:{len(test_sub)}")

    print("\n3. Model ve Tokenizer yükleniyor...")
    model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 3 sınıf var: contradiction (0), entailment (1), neutral (2)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    print("\n4. Metinler tokenize ediliyor...")
    # BERT modelinin girdileri doğru formatta işleyebilmesi için tokenize fonksiyonu
    def tokenize_function(examples):
        return tokenizer(
            examples["premise"], 
            examples["hypothesis"], 
            truncation=True,            # Belirlenen max_length'ten uzun metinleri keser
            padding="max_length",       # Kısa metinleri max_length boyutuna ulaşana kadar doldurur
            max_length=256              # Modelin tek seferde işleyeceği maksimum token sayısı
        )

    # Hazırlanan fonksiyon veri setindeki tüm satırlara toplu (batched) olarak uygula
    tokenized_train = train_sub.map(tokenize_function, batched=True)
    tokenized_test = test_sub.map(tokenize_function, batched=True)

    # PyTorch kütüphanesinin modeli eğitebilmesi için veri setinin formatı PyTorch Tensor'e çevirilir
    # Sadece modelin ihtiyaç duyduğu girdiler (id'ler, maskeler ve etiketler) korunur
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    print("\n5. Eğitim parametreleri ayarlanıyor...")
    training_args = TrainingArguments(
        output_dir="./sonuclar",          # Eğitim sırasında oluşan ara kontrol noktalarının (checkpoint) kaydedileceği dizin
        eval_strategy="epoch",            # Modelin başarısı (accuracy) her epoch (döngü) sonunda test edilir
        learning_rate=2e-5,               # Model ağırlıklarının güncellenme hızı (öğrenme katsayısı)
        per_device_train_batch_size=16,   # Ekran kartına (GPU) tek seferde beslenecek veri adedi
        num_train_epochs=3,               # Modelin tüm veri setini baştan sona kaç kez göreceği (döngü sayısı)
        weight_decay=0.01,                # Ezberlemeyi (overfitting) önlemek için uygulanan ağırlık azaltma regülasyonu
        save_strategy="epoch",            # Her döngü sonunda model durumunun diske kaydedilmesi
        load_best_model_at_end=True,      # Eğitim bittiğinde test verisinde en yüksek başarıyı göstermiş checkpoint'i geri yükler
        fp16=torch.cuda.is_available()    # GPU aktifse performansı artırmak ve VRAM tasarrufu sağlamak için yarım hassasiyet kullanır
    )

    # Modelin ham tahmin çıktılarını alıp başarı oranını hesaplayan fonksiyon
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # En yüksek olasılığa sahip sınıfın indeksi (0, 1 veya 2) tahmin olarak seçilir
        predictions = np.argmax(logits, axis=-1)
        # Doğru tahminlerin toplam tahminlere oranı hesaplanır
        acc = np.mean(predictions == labels)
        return {"accuracy": acc}

    print("\n6. Eğitim başlatılıyor...")
    # Eğitim sürecini yönetecek olan ana Hugging Face sınıflarının bağlanması
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    # Eğitimi başlatan komut
    trainer.train()

    print("\n7. Eğitilmiş model kaydediliyor...")
    # İleride LIME ve saldırı kodlarında doğrudan yükleyip kullanabilmek için eğitilen model ve tokenizer kaydedilir
    model.save_pretrained("./modeller/hukuk_bert_model")
    tokenizer.save_pretrained("./modeller/hukuk_bert_model")
    print("Model başarıyla './modeller/hukuk_bert_model' klasörüne kaydedildi!")

if __name__ == "__main__":
    main()