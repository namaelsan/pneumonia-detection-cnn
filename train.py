import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import kagglehub
import os

print("GPU Kullanılabilir mi?: ", len(tf.config.list_physical_devices('GPU')) > 0)

# 1. VERİ YOLUNU BULMA
print("Veri seti yolu bulunuyor...")
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print(f"Veri seti şurada: {path}")

# Veri seti içindeki klasör yapısı genelde şöyledir: path/chest_xray/train
base_dir = os.path.join(path, "chest_xray")
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
val_dir = os.path.join(base_dir, 'val')

# 2. VERİ ÖN İŞLEME (Data Augmentation)
# Eğitim verisini biraz "zorlaştırıyoruz" ki model ezberlemesin (Overfitting engelleme)
train_datagen = ImageDataGenerator(
    rescale=1./255,        # Renkleri 0-1 arasına sıkıştır
    rotation_range=20,     # Hafif döndür
    zoom_range=0.2,        # Yakınlaş/Uzaklaş
    horizontal_flip=True,  # Yatay çevir
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255) # Test verisine dokunmuyoruz, sadece normalize.

print("Görüntüler yükleniyor...")
# Görüntüleri klasörlerden okuyup modele besleyen "Generator"lar
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224), # VGG16 bu boyutu sever
    batch_size=32,
    class_mode='binary'     # Hasta mı / Değil mi? (0 veya 1)
)

validation_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# 3. TRANSFER LEARNING MODELİ (VGG16)
# ImageNet ağırlıklarını kullanıyoruz, en üst katmanı (include_top=False) almıyoruz.
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Baz modelin ağırlıklarını dondur (Eğitilmesin, hazır bilgi kullanılsın)
for layer in base_model.layers:
    layer.trainable = False

# Kendi sınıflandırma katmanımızı ekleyelim
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x) # Aşırı öğrenmeyi engelle
predictions = Dense(1, activation='sigmoid')(x) # Çıktı katmanı (0-1 arası değer)

model = Model(inputs=base_model.input, outputs=predictions)

# Modeli derle
model.compile(optimizer=Adam(learning_rate=0.0001), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.summary()

# 4. EĞİTİMİ BAŞLAT
# Bilgisayarın CPU kullanıyorsa bu işlem biraz uzun sürebilir.
# Hızlı sonuç için epoch sayısını düşük tuttum (3). İyi sonuç için 10 yapılabilir.
history = model.fit(
    train_generator,
    epochs=3, 
    validation_data=validation_generator
)

# 5. MODELİ KAYDET
model.save('pneumonia_model.keras')
print("Model 'pneumonia_model.keras' olarak kaydedildi!")

# 6. GRAFİKLERİ ÇİZ (Rapor İçin)
plt.figure(figsize=(12, 4))

# Accuracy Grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
plt.plot(history.history['val_accuracy'], label='Doğrulama Başarısı')
plt.legend()
plt.title('Başarı Oranı (Accuracy)')

# Loss Grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.legend()
plt.title('Kayıp (Loss)')

plt.savefig('egitim_grafigi.png')
print("Grafik 'egitim_grafigi.png' olarak kaydedildi.")