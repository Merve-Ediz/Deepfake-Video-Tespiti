import streamlit as st
import cv2
import torch
import torch.nn as nn
from PIL import Image # Pillow: görüntüyü PIL formatına çevirip işlemede kullanılır.
import numpy as np
from facenet_pytorch import MTCNN
import timm # timm = hazır model kütüphanesidir. Burada Xception mimarisini oluşturmak için kullanılır.

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Deepfake Tespit Sistemi", layout="wide")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- AKILLI MODEL YÜKLEME FONKSİYONU ---
@st.cache_resource # Streamlit “cache” dekoratörü. Bu fonksiyon bir kere çalışır, sonucu (modeli) saklar. Sayfada etkileşim olunca (butona basınca vs) modeli tekrar tekrar yüklemesin diyedir
def load_project_model():
    # 1. Xception Modelini Oluştur
    st.write("🛠️ Model iskeleti oluşturuluyor...")
    # num_classes=2 (Real/Fake)
    model = timm.create_model('xception', pretrained=False, num_classes=2)
    
    # 2. İndirdiğin dosyayı yükle
    model_path = "model.pth"  # Dosya adının bu olduğundan emin ol!
    
    try:
        # Ağırlıkları dosyadan oku
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Ağırlık sözlüğü boş başlatılıyor.
        state_dict = None
        if 'state_dict' in checkpoint: # Eğer checkpoint bir dict ve içinde 'state_dict' anahtarı varsa onu al, Eğer 'state_dict' yok ama 'model' varsa onu al.
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint # Direkt sözlük olabilir. “checkpoint direkt state_dict’in kendisiymiş” diye kabul eder.
            
        # --- KRİTİK KISIM: İSİM UYUŞMAZLIĞINI DÜZELTME ---
        # Dosyadaki anahtarlar (keys) ile modeldeki anahtarları eşleştiriyoruz.
        new_state_dict = {} # Amaç: checkpoint’teki anahtar isimlerini temizleyip, modelin beklediği isimlerle eşleştirerek buraya koymaktır.
        
        # Modelin beklediği tüm ağırlık anahtar isimleri listeleniyor. model.state_dict() = modelin parametrelerini “isim -> tensor” şeklinde döndürür.
        model_keys = list(model.state_dict().keys())
        
        # Checkpoint’teki her parametreyi dolaşıyor: k = parametre adı (string) v = o parametrenin tensörü (ağırlıklar)
        for k, v in state_dict.items():
            name = k # k’yı direkt kullanmak yerine name diye bir değişkende “düzenlenebilir kopya” tutuyor.
            
            # Yaygın önekleri temizle/kırp
            if name.startswith('module.'): name = name[7:]
            if name.startswith('model.'): name = name[6:]
            if name.startswith('net.'): name = name[4:]
            if name.startswith('backbone.'): name = name[9:] 
            
            # FC katmanı uyumsuzluğunu düzelt. Bazı modellerin son katmanı last_linear diye geçer. onu fc yapar. Böylece anahtar isimleri tutar.
            if "last_linear" in name:
                name = name.replace("last_linear", "fc")
            if "classifier" in name:
                name = name.replace("classifier", "fc")
                
            # Eğer temizlenmiş isim modelimizde varsa ekle
            if name in model_keys:
                new_state_dict[name] = v
                
        # Eğer hiç eşleşme olmadıysa hata ver
        if len(new_state_dict) == 0:
            st.error("🚨 HATA: Dosya yüklendi ama hiçbir katman eşleşmedi! Yanlış model mimarisi indirilmiş olabilir.")
            st.stop()
            
        # Ağırlıkları modele yükler. strict=False: “uyuşanları yükle, uyuşmayanları es geç” modudur.
        msg = model.load_state_dict(new_state_dict, strict=False)
        st.success(f"✅ Model Başarıyla Yüklendi! ({len(new_state_dict)} katman eşleşti)")
        
    except FileNotFoundError:
        st.error(f"❌ '{model_path}' dosyası bulunamadı! Klasörde dosyanın adının tam olarak 'model.pth' olduğundan emin ol.")
        st.stop()
    except Exception as e: # Yukarıdaki özel hata dışında herhangi bir hata olursa ekrana basar.
        st.error(f"❌ Model yükleme hatası: {e}")
        st.stop()

    # Modeli seçilen cihaza taşır.
    model.to(DEVICE)
    model.eval() # Modeli inference moduna alır. Video analizinde doğru davranış budur.
    return model

# --- YÜZ BULUCU ---
@st.cache_resource # Face detector da cache’leniyor. Çünkü MTCNN de yüklenirken zaman alabilir.
def load_face_detector():
    return MTCNN(keep_all=False, select_largest=True, device=DEVICE, margin=80) # keep_all=False: birden fazla yüz bulsa bile hepsini döndürme, tek yüz döndür.

# Modelleri Başlat
with st.spinner('Yapay Zeka Hazırlanıyor...'): # Ekranda “loading” gösterir.
    model = load_project_model() # Model kurulup ağırlıklar yüklenir. Cache sayesinde tekrar tekrar yüklenmez.
    mtcnn = load_face_detector()

# --- VİDEO İŞLEME ---
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = [] # Her analiz edilen kare için “fake olasılığı” buraya eklenecek.
    frames_processed = 0 # Kaç kare okundu sayacı
    
    # xception.yaml dosyasındaki verilere göre:
    # resolution: 256
    # mean: [0.5, 0.5, 0.5]
    # std: [0.5, 0.5, 0.5]
    
    from torchvision import transforms
    preprocess = transforms.Compose([ # Görüntüyü modele girmeden önce hazırlar.
        transforms.Resize((256, 256)),
        transforms.ToTensor(), # Değerleri genelde 0..1 aralığına getirir.
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    my_bar = st.progress(0) #Yüzdelik bar / %0
    debug_area = st.empty() # Sayfada “boş bir yer” açar. Sonra buraya debug metni basılır.
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Her 15 karede bir analiz et.
        if frames_processed % 15 == 0:
            try:
                # BGR -> RGB
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb) # MTCNN için NumPy RGB array’i PIL Image’e çevirir.
                
                # Yüz Bul
                face = mtcnn(img_pil)
                
                if face is not None:
                    # MTCNN çıktısını düzelt
                    face_np = face.permute(1, 2, 0).cpu().numpy() # MTCNN çıktısı çoğu zaman C x H x W'dir. Görüntü gibi kullanmak için H x W x C lazım
                    face_np = (face_np - face_np.min()) / (face_np.max() - face_np.min()) # Face tensörünü 0..1 aralığına normalize ediyor. Amaç: tekrar “görüntü” gibi PIL’e çevirebilmektir.
                    face_pil = Image.fromarray((face_np * 255).astype(np.uint8)) # 0..1 → 0..255’e çevir (*255) Görüntü dosyası gibi olması için 8-bit tamsayı
                    
                    # Modele Hazırla (256x256 olacak)
                    input_tensor = preprocess(face_pil)
                    input_batch = input_tensor.unsqueeze(0).to(DEVICE) # Batch: (B,C,H,W) unsqueeze(0): 0. boyuta 1 ekler → batch size = 1 Yani (C,H,W) → (1,C,H,W)
                    
                    with torch.no_grad(): # Inference sırasında gradient hesaplanmasın: daha az bellek, daha hızlı
                        output = model(input_batch)
                        probs = torch.nn.functional.softmax(output, dim=1) # Logits’i olasılığa çevirir.
                        
                        # Class 0 = Real, Class 1 = Fake
                        prob_real = probs[0][0].item() # .item() tensör içindeki sayıyı Python float yapar. Olasılık verir.
                        prob_fake = probs[0][1].item()
                        
                        # Skor olarak Fake ihtimalini (Class 1) kaydedelim
                        predictions.append(prob_fake)
                        
                        # CANLI TAKİP:
                        debug_area.code(f"""
                        Kare {frames_processed}:
                        Class 0 (Real) İhtimali: {prob_real:.4f}
                        Class 1 (Fake) İhtimali: {prob_fake:.4f}
                        """)
                        
            except Exception as e:
                pass

        # Kare sayacını 1 artır. 
        frames_processed += 1
        if frames_processed < 300: # İlk 300 kare için progress bar güncellemesi yapacak. 300 kareyi “%100 gibi” kabul eder.
            my_bar.progress(frames_processed / 300)
            
    cap.release()
    my_bar.empty()
    debug_area.empty()
    return predictions

# --- ARAYÜZ ---
st.title("🕵️‍♂️ Deepfake Video Analiz (FF++)")

uploaded_file = st.file_uploader("Video Yükle", type=['mp4', 'avi'])

if uploaded_file is not None:
    with open("temp.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer()) # Streamlit’in yüklenen dosyasının içeriğini alıp temp.mp4 içine yazıyor. getbuffer() = dosyanın byte verisini verir.
        
    st.video(uploaded_file) # Yüklenen videoyu sayfada oynatılabilir şekilde gösterir.
    
    if st.button("ANALİZ ET"):
        preds = process_video("temp.mp4") # preds = fake olasılıkları listesidir. fake'ler toplanır, ortalaması alınır. oranı 0.50'yi geçerse fake der, geçmezse 1'den çıkarılarak real der.
        
        if len(preds) == 0:
            st.warning("Yüz bulunamadı.")
        else:
            avg_score = sum(preds) / len(preds)
            
            # Eşik Değeri
            if avg_score > 0.50:
                st.error(f"🚨 FAKE (SAHTE) - Güven: %{avg_score*100:.1f}📈")
            else:
                st.success(f"✅ REAL (GERÇEK) - Güven: %{(1-avg_score)*100:.1f}📈")