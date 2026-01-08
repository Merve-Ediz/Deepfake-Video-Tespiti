import os
import sys
import time
import glob # Klasörlerdeki uygun dosyaları bulma aracıdır.
import cv2
import dlib
import yaml
import logging
import concurrent.futures # Birden fazla videoyu aynı anda işlemek için kullanılır.
import numpy as np
from tqdm import tqdm
from pathlib import Path
from imutils import face_utils # Landmark çıktısını numpy array'e çevirmeyi kolaylaştırır.
from skimage import transform as trans # Yüzü döndür, ölçekle, hizala gibi işlemler için kullanılır. 


# -------------------------
# Logging
# -------------------------
def create_logger(log_path: str):
    logger = logging.getLogger("preprocess")
    logger.setLevel(logging.INFO) #Loglarda Warning, Error, Info bilgilerini yazar.
    logger.handlers.clear() # Script tekrar çalıştırılırsa üst üste eklenmesini engeller.

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    fh = logging.FileHandler(log_path)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh) # Logları dosyaya yazan parçayı logger'a bağlar.

    sh = logging.StreamHandler() # Logları ekrana yazar.
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# -------------------------
# Landmark helpers
# -------------------------
def get_keypts(image, face, predictor): # Amaç landmark predictor ile 5 ana noktayı alıp numpy array olarak döndürmektir. 
    shape = predictor(image, face) # Predictor'e bu görüntüde şu yüz bölgesi var demektir.

    leye = np.array([shape.part(37).x, shape.part(37).y]).reshape(-1, 2) # Göz, burun ağız bölgelerinden bir nokta seçilip bunu (1,2) şekline sokar.
    reye = np.array([shape.part(44).x, shape.part(44).y]).reshape(-1, 2)
    nose = np.array([shape.part(30).x, shape.part(30).y]).reshape(-1, 2)
    lmouth = np.array([shape.part(49).x, shape.part(49).y]).reshape(-1, 2)
    rmouth = np.array([shape.part(55).x, shape.part(55).y]).reshape(-1, 2)

    return np.concatenate([leye, reye, nose, lmouth, rmouth], axis=0)


def extract_aligned_face_dlib(face_detector, predictor, image_bgr, res=256, mask_bgr=None):
    """
    - Dlib ile yüzü bulur
    - 5 nokta landmark ile SimilarityTransform uygular (yüz yatık/ters ise hizalar)
    - res x res olarak kırpar
    - ayrıca istenirse aligned yüz için 68-point landmark da döndürür
    """
    def img_align_crop(img_rgb, landmark, outsize, scale=1.3, mask=None): # Üst fonksiyonda kullanılacak yardımcı fonksiyonları tanımlar.
        target_size = [112, 112] # Landmark şablonu
        dst = np.array(
            [
                [30.2946, 51.6963],
                [65.5318, 51.5014],
                [48.0252, 71.7366],
                [33.5493, 92.3655],
                [62.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        if target_size[1] == 112:
            dst[:, 0] += 8.0 # Hedef noktaları x ekseninde 8 birim sağa kaydırır. Standart ayardır.

        dst[:, 0] = dst[:, 0] * outsize[0] / target_size[0] # x koordinatlarını 112 → 256 ölçeğine orantılı büyütür.
        dst[:, 1] = dst[:, 1] * outsize[1] / target_size[1]

        margin_rate = scale - 1 # scale=1.3 ise margin_rate=0.3'tür. Yani “%30 ekstra boşluk/pay” demektir.
        x_margin = outsize[0] * margin_rate / 2.0 # Toplam margin %30 iken, bir tarafa düşen yarısını bulur.
        y_margin = outsize[1] * margin_rate / 2.0

        dst[:, 0] += x_margin # Hedef noktaları x’te margin kadar iter. Pay eklemenin parçasıdır.
        dst[:, 1] += y_margin

        dst[:, 0] *= outsize[0] / (outsize[0] + 2 * x_margin) # Margin ile büyümüş alanı tekrar normalize eder.
        dst[:, 1] *= outsize[1] / (outsize[1] + 2 * y_margin)

        src = landmark.astype(np.float32)

        tform = trans.SimilarityTransform() # SimilarityTransform: döndürme, ölçekleme, kaydırma yapabilir.
        tform.estimate(src, dst) # Yüz yamuksa düzelt, konumu ayarla” burada hesaplanır.
        M = tform.params[0:2, :] # tform.params = 3x3 transform matrisidir. OpenCV warpAffine 2x3 ister.

        img_aligned = cv2.warpAffine(img_rgb, M, (outsize[1], outsize[0]))
        img_aligned = cv2.resize(img_aligned, (outsize[1], outsize[0])) # affine transform'un ölçeklendirdiği resmi netleştirir.

        if mask is not None:
            mask_aligned = cv2.warpAffine(mask, M, (outsize[1], outsize[0]))
            mask_aligned = cv2.resize(mask_aligned, (outsize[1], outsize[0]))
            return img_aligned, mask_aligned

        return img_aligned, None

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb, 1) # Dlib yüz detektörünü çalıştırır.
    if len(faces) == 0:
        return None, None, None

    # en büyük yüz
    face = max(faces, key=lambda r: r.width() * r.height())
    key5 = get_keypts(rgb, face, predictor) # Seçilen yüz için 5 landmark noktası çıkarılır → key5 (5x2 array).

    if mask_bgr is not None:
        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
        cropped_rgb, mask_out = img_align_crop(rgb, key5, outsize=(res, res), mask=mask_rgb)
    else:
        cropped_rgb, mask_out = img_align_crop(rgb, key5, outsize=(res, res), mask=None)

    cropped_bgr = cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR) # predictor bgr istediği için bgr'e çevrilir.

    # aligned yüz üzerinde 68 landmark
    faces2 = face_detector(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB), 1)
    if len(faces2) == 0:
        return None, None, None

    lm68 = predictor(cropped_bgr, faces2[0]) # Landmark çıkarılır.
    lm68 = face_utils.shape_to_np(lm68) # numpy array’e çevirir.

    return cropped_bgr, lm68, mask_out


# -------------------------
# Video processing
# -------------------------
def process_one_video(
    video_path: Path,
    save_root: Path,
    face_detector,
    face_predictor,
    mode: str,
    num_frames: int,
    stride: int,
    res: int,
    logger
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Video açılamadı: {video_path}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Videodaki toplam frame sayısını alır.
    if frame_count <= 0:
        logger.warning(f"Frame sayısı okunamadı: {video_path}")
        cap.release()
        return

    if mode == "fixed_num_frames":
        frame_idxs = np.linspace(0, frame_count - 1, num_frames, endpoint=True, dtype=int) # start ile stop arasında eşit aralıklı num tane frame indeksleri üretilir.
        frame_idxs = set(frame_idxs.tolist())
    elif mode == "fixed_stride":
        frame_idxs = set(np.arange(0, frame_count, stride, dtype=int).tolist())
    else:
        raise ValueError("mode fixed_num_frames veya fixed_stride olmalı")

    out_frames_dir = save_root / "frames" / video_path.stem
    out_land_dir = save_root / "landmarks" / video_path.stem
    out_frames_dir.mkdir(parents=True, exist_ok=True)
    out_land_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(frame_count): # 0’dan frame_count-1’e kadar tüm frame’leri sırayla dolaşır.
        ret, frame = cap.read() #  ret: True/False (okuma başarılı mı?) frame: görüntü

        if not ret:
            break
        if idx not in frame_idxs:
            continue

        cropped, lm68, _ = extract_aligned_face_dlib(
            face_detector=face_detector,
            predictor=face_predictor,
            image_bgr=frame,
            res=res,
            mask_bgr=None
        )
        if cropped is None or lm68 is None: # Eğer yüz çıkarma başarısız olduysa bu frame’i kaydetmeden geç.
            continue

        cv2.imwrite(str(out_frames_dir / f"{idx:03d}.png"), cropped) # idx:03d: idx sayısını 3 basamaklı yaz (5 → 005)
        np.save(str(out_land_dir / f"{idx:03d}.npy"), lm68)

    cap.release()


def preprocess_folder(
    videos_dir: Path,
    out_dir: Path,
    face_detector,
    face_predictor,
    mode: str,
    num_frames: int,
    stride: int,
    res: int,
    logger
):
    video_list = sorted([Path(p) for p in glob.glob(str(videos_dir / "**/*.mp4"), recursive=True)]) # glob klasör altındaki tüm mp4’leri listeler.
    if len(video_list) == 0:
        logger.warning(f"Video bulunamadı: {videos_dir}")
        return

    logger.info(f"{len(video_list)} video bulundu: {videos_dir}")

    start_time = time.monotonic()
    max_workers = max(1, os.cpu_count() or 1) # Thread sayısı ayarlar. os.cpu_count() CPU çekirdek sayısını döndürür.

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex: # Thread havuzu açar. with ... as ex: İş bittiğinde otomatik kapatır/temizler.
        futures = [] # Başlatılan işlerin takip nesneleridir.
        for vp in video_list:
            futures.append( # Her video path için aşağıda başlattığı işi listeye ekleyecek.
                ex.submit( # process_one_video fonksiyonunu ayrı bir thread’de çalıştır.
                    process_one_video,
                    vp, out_dir,
                    face_detector, face_predictor,
                    mode, num_frames, stride, res, logger
                )
            )

        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures)): # İlerleme çubuğu için toplam iş sayısı verilir.
            pass

    duration_min = (time.monotonic() - start_time) / 60.0
    logger.info(f"Bitti. Süre: {duration_min:.2f} dk")


# -------------------------
# Main (FF++)
# -------------------------
if __name__ == "__main__":
    # Config
    with open("./config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pp = cfg["preprocess"] # Kısaltma: Sürekli cfg["preprocess"] yazmak yerine pp diyor.
    dataset_name = pp["dataset_name"]["default"]
    dataset_root = Path(pp["dataset_root_path"]["default"])
    comp = pp["comp"]["default"]
    mode = pp["mode"]["default"]
    stride = int(pp["stride"]["default"])
    num_frames = int(pp["num_frames"]["default"])
    res = int(pp["output_res"]["default"])
    predictor_path = pp["shape_predictor_path"]["default"]

    if dataset_name != "FaceForensics++":
        raise ValueError("Bu sadeleştirilmiş script sadece FaceForensics++ içindir.")

    # Logger
    logger = create_logger(f"./logs/preprocess_{dataset_name}_{comp}.log")

    # Dlib models
    if not os.path.exists(predictor_path):
        logger.error(f"shape predictor yok: {predictor_path}")
        sys.exit(1)

    face_detector = dlib.get_frontal_face_detector() # Dlib’in hazır “frontal face detector”ünü oluşturur
    face_predictor = dlib.shape_predictor(predictor_path) # .dat dosyasından shape predictor yükler.

    # Dataset paths (ONLY actors originals + selected manipulations)
    ff_root = dataset_root / "FaceForensics++"

    # 1) Originals: actors only
    orig_actors_videos = ff_root / "original_sequences" / "actors" / comp / "videos"
    orig_actors_out = ff_root / "original_sequences" / "actors" / comp

    preprocess_folder(
        videos_dir=orig_actors_videos,
        out_dir=orig_actors_out,
        face_detector=face_detector,
        face_predictor=face_predictor,
        mode=mode,
        num_frames=num_frames,
        stride=stride,
        res=res,
        logger=logger
    )

    # 2) Manipulated sequences
    methods = pp["manipulated_methods"]["default"]
    for m in methods:
        vid_dir = ff_root / "manipulated_sequences" / m / comp / "videos"
        out_dir = ff_root / "manipulated_sequences" / m / comp

        preprocess_folder(
            videos_dir=vid_dir,
            out_dir=out_dir,
            face_detector=face_detector,
            face_predictor=face_predictor,
            mode=mode,
            num_frames=num_frames,
            stride=stride,
            res=res,
            logger=logger
        )

    logger.info("Preprocess tamamlandı.")
