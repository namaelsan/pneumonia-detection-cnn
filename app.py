import gradio as gr
import tensorflow as tf
import numpy as np
import os
import random
import shutil

# 1. MODELİ YÜKLE
print("Model yükleniyor...")
model_path = 'pneumonia_model.keras'
if not os.path.exists(model_path):
    model_path = 'pneumonia_model.h5'

try:
    model = tf.keras.models.load_model(model_path)
    print(f"Model başarıyla yüklendi: {model_path}")
except Exception as e:
    print(f"Model yüklenemedi! Hata: {e}")
    model = None 

# 2. ÖRNEKLERİ HAZIRLA
def ornekleri_hazirla_ve_getir():
    target_dir = "hazir_ornekler"
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    if not os.listdir(target_dir):
        print("Klasör boş, kaynaklardan örnekler kopyalanıyor...")
        base_dir = os.path.expanduser("~/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/test")
        if not os.path.exists(base_dir):
            base_dir = os.path.expanduser("~/.cache/kagglehub/datasets/paultimothymooney/chest-xray-pneumonia/versions/2/chest_xray/chest_xray/test")

        if os.path.exists(base_dir):
            normal_src = os.path.join(base_dir, "NORMAL")
            pneumonia_src = os.path.join(base_dir, "PNEUMONIA")
            
            def copy_random_files(src_dir, label_prefix, count=3):
                if os.path.exists(src_dir):
                    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if files:
                        picks = random.sample(files, min(count, len(files)))
                        for filename in picks:
                            new_filename = f"{label_prefix}_{filename}"
                            shutil.copy2(os.path.join(src_dir, filename), os.path.join(target_dir, new_filename))

            copy_random_files(normal_src, "saglikli", 3)
            copy_random_files(pneumonia_src, "hasta", 3)

    examples = []
    if os.path.exists(target_dir):
        files = os.listdir(target_dir)
        files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for filename in files:
            full_path = os.path.join(target_dir, filename)
            if "HASTA" in filename:
                etiket = "PNEUMONIA (Hasta/Zatürre)"
            elif "SAGLIKLI" in filename:
                etiket = "NORMAL (Sağlıklı)"
            else:
                etiket = "Bilinmiyor"
            examples.append([full_path, etiket])
            
    return examples

gradio_examples = ornekleri_hazirla_ve_getir()

# 3. TAHMİN FONKSİYONU
def teshis_koy(img, etiket_text):
    # etiket_text burada kullanılmıyor ama gradio inputs listesiyle eşleşmesi için parametre olarak kalmalı.
    if model is None:
        return {"Hata": 0}
    if img is None:
        return None
        
    img = np.array(img)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    img_array = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    
    confidence_pneumonia = float(prediction)
    confidence_normal = 1.0 - confidence_pneumonia
    
    sonuc_dict = {
        "Zatürre (Pneumonia)": confidence_pneumonia,
        "Normal (Sağlıklı)": confidence_normal
    }
    
    return sonuc_dict

# 4. ARAYÜZ TASARIMI
theme = gr.themes.Soft(primary_hue="blue", secondary_hue="slate")

with gr.Blocks(theme=theme, title="Zatürre Teşhis Asistanı") as demo:
    gr.Markdown(
        """
        # 🏥 Yapay Zeka Destekli Zatürre Teşhisi
        X-Ray görüntülerini analiz ederek zatürre riskini hesaplar.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            # GİRDİLER
            img_input = gr.Image(type="pil", label="X-Ray Görüntüsü", sources=["upload", "clipboard"])
            
            # --- DEĞİŞİKLİK BURADA ---
            # Textbox'ı tanımladık ama 'visible=False' yaptık.
            # Böylece ekranda yer kaplamaz ama veriyi tutabilir.
            lbl_true_label = gr.Textbox(visible=False) 
            
            with gr.Row():
                # ClearButton'dan lbl_true_label'ı kaldırdık (zaten gizli)
                btn_clear = gr.ClearButton(components=[img_input], value="Temizle")
                btn_run = gr.Button("Analiz Et", variant="primary")
        
        with gr.Column(scale=1):
            # ÇIKTI
            lbl_output = gr.Label(num_top_classes=2, label="Yapay Zeka Tahmini")

    # BUTON AKSİYONU
    # lbl_true_label hala input olarak verilmeli çünkü fonksiyon 2 parametre bekliyor
    btn_run.click(fn=teshis_koy, inputs=[img_input, lbl_true_label], outputs=[lbl_output])
    
    # ÖRNEKLER TABLOSU
    if gradio_examples:
        gr.Examples(
            examples=gradio_examples,       
            # inputs listesinde gizli textbox olduğu için 
            # Gradio metni oraya gönderir ama kullanıcı görmez.
            # Ancak örnekler listesinde metin görünmeye devam eder.
            inputs=[img_input, lbl_true_label], 
            outputs=[lbl_output],
            fn=teshis_koy,     
            run_on_click=True, 
            cache_examples=False,
            label="Hazır Örnekler (Tıklayın)"
        )

if __name__ == "__main__":
    demo.launch(share=True)