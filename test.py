import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# ---------------------------------------------------------
# 1. AYARLAR VE VERİ YOLU
# ---------------------------------------------------------
# Eğer app.py'de indirdiysen zaten cache'de duruyordur.
# Yine de garantilemek için aynı indirme/bulma mantığını kullanıyoruz.

print("Veri seti yolu bulunuyor...")
try:
    path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
    # Olası yollar
    potential_path_1 = os.path.join(path, "chest_xray", "test")
    potential_path_2 = os.path.join(path, "chest_xray", "chest_xray", "test")
    
    if os.path.exists(potential_path_2):
        test_dir = potential_path_2
    elif os.path.exists(potential_path_1):
        test_dir = potential_path_1
    else:
        raise Exception("Test klasörü bulunamadı!")
        
    print(f"Test verisi şuradan okunacak: {test_dir}")

except Exception as e:
    # Eğer kagglehub hata verirse manuel yol (app.py'deki gibi proje içine indirdiysen)
    print("Kagglehub hatası veya manuel yol kontrolü...")
    test_dir = "./veri_deposu/chest_xray/test" # Eğer app.py ile buraya indirdiysen
    if not os.path.exists(test_dir):
        # Son çare manuel yolunu yazabilirsin
        print("HATA: Test verisi bulunamadı. Lütfen test_dir değişkenini elle düzelt.")
        exit()

# ---------------------------------------------------------
# 2. MODELİ YÜKLE
# ---------------------------------------------------------
model_path = 'pneumonia_model.keras'
if not os.path.exists(model_path):
    model_path = 'pneumonia_model.h5'

print(f"Model yükleniyor: {model_path}")
model = tf.keras.models.load_model(model_path)

# ---------------------------------------------------------
# 3. TEST VERİSİNİ HAZIRLA
# ---------------------------------------------------------
test_datagen = ImageDataGenerator(rescale=1./255)

print("Test görüntüleri yükleniyor...")
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False  # ÇOK ÖNEMLİ: Confusion Matrix için sıranın bozulmaması lazım!
)

# ---------------------------------------------------------
# 4. TAHMİN YAP
# ---------------------------------------------------------
print("Tahminler yapılıyor (Bu işlem biraz sürebilir)...")
# Modelden olasılıkları al (0 ile 1 arası değerler)
predictions = model.predict(test_generator, steps=len(test_generator))

# Olasılıkları sınıfa çevir (0.5'ten büyükse 1, değilse 0)
y_pred = (predictions > 0.5).astype(int).flatten()

# Gerçek etiketleri al
y_true = test_generator.classes

# Etiket isimleri (Alfabetik sıra: 0=NORMAL, 1=PNEUMONIA)
class_names = ['NORMAL', 'PNEUMONIA']

# ---------------------------------------------------------
# 5. METRİKLERİ HESAPLA VE YAZDIR
# ---------------------------------------------------------
print("\n" + "="*50)
print("MODEL DEĞERLENDİRME RAPORU")
print("="*50)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix (Karmaşıklık Matrisi):")
print(cm)

# Classification Report (Precision, Recall, F1-Score)
print("\nSınıflandırma Raporu:")
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# Genel F1 Score
f1 = f1_score(y_true, y_pred)
print(f"\nGenel F1-Score: {f1:.4f}")

# ---------------------------------------------------------
# 6. GÖRSELLEŞTİR VE KAYDET
# ---------------------------------------------------------
# Confusion Matrix'i görselleştir
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Gerçek Etiket (Ground Truth)')
plt.xlabel('Tahmin Edilen (Predicted)')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("\nGRAFİK KAYDEDİLDİ: 'confusion_matrix.png'")