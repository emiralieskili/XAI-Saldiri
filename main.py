import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np

def encode_labels(example):
    # Metinsel etiketleri sayısal değerlere dönüştürüyoruz
    mapping = {"contradiction": 0, "entailment": 1, "neutral": 2}
    # Eğer etiket veri setinde beklenmeyen bir formatta ise varsayılan olarak 2 (neutral) veriyoruz
    example["label"] = mapping.get(str(example["label"]).strip().lower(), 2)
    return example

def main():
    print("1. Cihaz kontrol ediliyor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Kullanılan Cihaz: {device.upper()}")

    print("\n2. Veri seti yükleniyor...")
    dataset = load_dataset("Turkish-NLI/legal_nli_TR_V1")
    
    # Geliştirme aşamasında eğitimin hızlı bitmesi için verinin bir kısmını alıyoruz
    train_sub = dataset["train"].select(range(1000)).map(encode_labels)
    test_sub = dataset["test"].select(range(200)).map(encode_labels)

    print("\n3. Model ve Tokenizer yükleniyor...")
    model_name = "dbmdz/bert-base-turkish-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 3 sınıfımız var: contradiction (0), entailment (1), neutral (2)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    print("\n4. Metinler tokenize ediliyor...")
    def tokenize_function(examples):
        return tokenizer(
            examples["premise"], 
            examples["hypothesis"], 
            truncation=True, 
            padding="max_length", 
            max_length=256
        )

    tokenized_train = train_sub.map(tokenize_function, batched=True)
    tokenized_test = test_sub.map(tokenize_function, batched=True)

    # PyTorch Tensor formatı ayarı (Eğitim sırasındaki veri tipi hatalarını önler)
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    print("\n5. Eğitim parametreleri ayarlanıyor...")
    training_args = TrainingArguments(
        output_dir="./sonuclar",          # Eğilen modelin geçici kaydedileceği yer
        eval_strategy="epoch",            # Her döngüde başarıyı test et
        learning_rate=2e-5,               # Öğrenme katsayısı
        per_device_train_batch_size=8,    # Cihazı yormamak için küçük batch boyutu
        num_train_epochs=2,               # Eğitimin döngü sayısı
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,      # (Hata buradaydı, sonuna virgül eklendi)
        fp16=torch.cuda.is_available()    # GPU aktifse eğitimi hızlandırmak için FP16 kullanır
    )

    # Başarı metriğini hesaplayan basit bir fonksiyon
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = np.mean(predictions == labels)
        return {"accuracy": acc}

    print("\n6. Eğitim başlatılıyor...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n7. Eğitilmiş model kaydediliyor...")
    model.save_pretrained("./modeller/hukuk_bert_model")
    tokenizer.save_pretrained("./modeller/hukuk_bert_model")
    print("Model başarıyla './modeller/hukuk_bert_model' klasörüne kaydedildi!")

if __name__ == "__main__":
    main()