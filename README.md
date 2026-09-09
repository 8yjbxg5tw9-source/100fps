# 720p → 8K @ 1000 FPS — Lokal Video Pipeline

> **Məqsəd:** 720p 30/60 FPS videoları lokal kompüterdə **8K (7680×4320)** rezolusiya
> və **1000 FPS** kadr tezliyinə qaldıran 10 addımlıq pipeline.

## Pipeline planı (10 addım)

| Addım | Adı | Status |
|-------|-----|--------|
| Step 1 | Sistem mühitinin inisializasiyası, avadanlıq analizi, mərkəzi konfiqurasiya | ✅ hazır |
| Step 2 | Video analizi, audio çıxarılması, kadrlara bölünmə | ✅ hazır |
| Step 3 | Optik axın / kadr interpolasiyası (→1000 FPS, RIFE) | ⏳ növbəti |
| Step 4 | Super-rezolusiya (→8K, tiled inference) | ⬜ |
| Step 5 | Denoise / deblur / rəng bərpası | ⬜ |
| Step 6 | Kadrların keyfiyyət yoxlaması və filtrasiyası | ⬜ |
| Step 7 | 8K kadrların videoya yığılması (FFmpeg encode) | ⬜ |
| Step 8 | Audio sinxronizasiya və köçürmə | ⬜ |
| Step 9 | Metadata, HDR, çıxış optimallaşdırması | ⬜ |
| Step 10 | Təmizlik, hesabat və бенчмарк | ⬜ |

---

## Step 1 — Sistem Mühiti və Konfiqurasiya

Step 1 proqram işə düşəndə bunları edir:

1. **Sistem yoxlaması** — `FFmpeg` mövcudluğu (`PATH`-də axtarış + versiya oxuma);
   yoxdursa dəqiq quraşdırma təlimatı ilə xəbərdarlıq verir.
2. **Python asılılıqları** — `torch`, `torchvision`, `opencv-python`, `numpy`,
   `Pillow`, `tqdm` yoxlanılır; çatışmayanlar `pip` ilə avtomatik qurulur.
3. **GPU/VRAM analizi** — NVIDIA CUDA yoxlanılır, VRAM həcmi GB ilə hesablanır,
   VRAM-a görə `tile_size` avtomatik seçilir:

   | VRAM | `tile_size` | Səbəb |
   |------|-------------|-------|
   | < 8 GB (və ya CPU rejimi) | `256` | Çökmədən qorunmaq üçün kiçik hissələr |
   | 8 – 16 GB | `512` | Balanslı rejim |
   | > 16 GB | `1024` | Maksimum sürət |

4. **Mərkəzi konfiqurasiya** — giriş/müvəqqəti/çıxış yolları, hədəf FPS
   (`1000.0`) və hədəf ölçü (`7680×4320`) tək `PipelineConfig` obyektində
   saxlanılır; Step 2-də birbaşa istifadə olunur.
5. **İşçi qovluqlar** — `temp_raw_frames`, `interpolated_720p`, `upscaled_8k`
   avtomatik yaradılır.
6. **Loglama** — hər addım vaxt damğası ilə terminala yazılır,
   giriş videosu tapılmazsa `FileNotFoundError` ilə icra dayanır.

### Quraşdırma

```bash
# FFmpeg (mütləqdir — video kadrlara bölmək/yığmaq üçün)
# Ubuntu/Debian:
sudo apt update && sudo apt install -y ffmpeg
# macOS:
brew install ffmpeg
# Windows (PowerShell):
winget install Gyan.FFmpeg

# Python asılılıqları (avtomatik də qurulur, əl ilə istəsəniz):
pip install -r requirements.txt
```

> CUDA-lı PyTorch üçün [pytorch.org](https://pytorch.org/get-started/locally/)
> səhifəsindəki CUDA-ya uyğun `pip` əmrini işlədin. Step 1 CUDA tapmasa
> avtomatik CPU rejiminə keçir (8K emal CPU-da çox yavaş olacaq).

### İstifadə

```bash
# Minimal:
python main.py --input video/input_720p.mp4 --output video/output_8k_1000fps.mp4

# Tam:
python main.py \
  --input  video/input_720p.mp4 \
  --output video/output_8k_1000fps.mp4 \
  --workspace workspace \
  --target-fps 1000 \
  --target-width 7680 --target-height 4320

# Avtomatik pip quraşdırmanı söndürmək:
python main.py --input in.mp4 --output out.mp4 --no-auto-install

# tile_size-i əl ilə təyin etmək (avtomatik VRAM məntiqini əvəz edir):
python main.py --input in.mp4 --output out.mp4 --tile-size 512
```

Uğurlu işə düşmə nümunəsi:

```text
[2026-09-09 14:55:01] [INFO] === Step 1: Environment initialization started ===
[2026-09-09 14:55:01] [INFO] FFmpeg detected: ffmpeg version 6.0 (/usr/bin/ffmpeg)
[2026-09-09 14:55:01] [INFO] Dependency OK: torch
[2026-09-09 14:55:01] [INFO] Dependency OK: torchvision
...
[2026-09-09 14:55:02] [INFO] CUDA detected: NVIDIA GeForce RTX 3080 (10.0 GB VRAM) -> tile_size=512
[2026-09-09 14:55:02] [INFO] Created directory: workspace/temp_raw_frames
[2026-09-09 14:55:02] [INFO] Created directory: workspace/interpolated_720p
[2026-09-09 14:55:02] [INFO] Created directory: workspace/upscaled_8k
[2026-09-09 14:55:02] [INFO] Step 1 completed. Config ready for Step 2.
```

Konfiqurasiya avtomatik olaraq `workspace/config.json` faylına da yazılır —
Step 2 onu birbaşa oxuyub işlədə bilər:

```python
from pipeline.config import PipelineConfig

config = PipelineConfig.load("workspace/config.json")
print(config.temp_raw_frames)   # Step 2-nin çıxış qovluğu
print(config.tile_size)         # Step 4-ün istifadə edəcəyi dəyər
```

### Proqramlı istifadə (digər AI / modul kimi)

```python
from pipeline.step01_environment import setup_environment

config = setup_environment(
    input_video_path="video/input_720p.mp4",
    final_output_path="video/output_8k_1000fps.mp4",
)
# config.device      -> "cuda" və ya "cpu"
# config.vram_gb     -> məs: 10.0
# config.tile_size   -> 256 / 512 / 1024
```

## Step 2 — Video Analizi, Audio və Kadr Çıxarılması

Step 1-dən gələn `config` obyektini qəbul edir və Step 3-ə (RIFE interpolasiyası)
hazır vəziyyətə gətirir:

1. **Meta-məlumatlar** — FFprobe (yoxdursa OpenCV fallback) ilə ölçü, orijinal
   FPS, ümumi kadr sayı, müddət oxunur; 1000 FPS-ə çatmaq üçün artım əmsalı
   hesablanır (məs: 30 → 1000 FPS = **33.33x**). Nəticə həm atribut, həm də
   `metadata` dictionary kimi `config`-ə yazılır.
2. **Audio** — səs treki itkisiz `workspace/input_audio.wav` kimi ayrılır
   (`--audio-format aac` ilə AAC də olar). Səs yoxdursa log qeyd olunur və
   xətasız davam edilir (Step 8 səssiz çıxış yaradacaq).
3. **Kadrlar** — bütün kadrlar itkisiz `frame_%06d.png` formatında
   `temp_raw_frames`-ə çıxarılır (`--image-format jpg` ilə kiçik həcm də olar),
   proses `tqdm` progress bar ilə göstərilir.
4. **Validasiya** — çıxarılan fayl sayı ilə orijinal kadr sayı müqayisə olunur;
   uyğunsuzluq xəbərdarlıq verir (çökmür).

```bash
# Step 1 + Step 2 birlikdə:
python main.py --input video/in.mp4 --output video/out.mp4

# Yalnız Step 1 (mühit + konfiq):
python main.py --input video/in.mp4 --output video/out.mp4 --step1-only

# Saxlanmış konfiqdən Step 2-ni təkrar işə salmaq:
python main.py --config workspace/config.json

# JPG kadrlar + AAC səs (daha az disk yeri):
python main.py --input video/in.mp4 --output video/out.mp4 \
  --image-format jpg --audio-format aac
```

Exit kodları: `0` uğur, `1` video tapılmadı, `2` digər xəta,
`3` FFmpeg tapılmadı (Step 2-dən etibarən məcburidir).

### Proqramlı istifadə (Step 2)

```python
from pipeline.config import PipelineConfig
from pipeline.step02_frames import Step02Frames

config = PipelineConfig.load("workspace/config.json")
result = Step02Frames(image_format="png", audio_format="wav").run(config)

print(result.interpolation_factor)  # məs: 33.33
print(result.frame_dir)             # Step 3-ün giriş qovluğu
print(result.audio_path)            # Step 8-in istifadə edəcəyi səs (və ya None)
```

### Layihə strukturu

```text
100fps/
├── main.py                      # CLI giriş nöqtəsi (Step 1 + Step 2)
├── requirements.txt             # Python asılılıqları
├── pipeline/
│   ├── __init__.py
│   ├── base.py                  # PipelineStep abstrakt bazası (bütün 10 addım üçün)
│   ├── config.py                # PipelineConfig — mərkəzi konfiqurasiya obyekti
│   ├── logger.py                # Vaxt damğalı terminal logları
│   ├── exceptions.py            # Xüsusi xəta tipləri
│   ├── step01_environment.py    # STEP 1: mühit + GPU + konfiqurasiya
│   └── step02_frames.py         # STEP 2: analiz + audio + kadr çıxarılması
└── tests/
    ├── test_step01_environment.py
    └── test_step02_frames.py
```

### Testlər

```bash
pip install pytest
pytest tests/ -v
```
