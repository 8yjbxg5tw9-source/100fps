# 720p → 8K @ 1000 FPS — Lokal Video Pipeline

> **Məqsəd:** 720p 30/60 FPS videoları lokal kompüterdə **8K (7680×4320)** rezolusiya
> və **1000 FPS** kadr tezliyinə qaldıran 10 addımlıq pipeline.

## Pipeline planı (10 addım)

| Addım | Adı | Status |
|-------|-----|--------|
| Step 1 | Sistem mühitinin inisializasiyası, avadanlıq analizi, mərkəzi konfiqurasiya | ✅ hazır |
| Step 2 | Video analizi, audio çıxarılması, kadrlara bölünmə | ✅ hazır |
| Step 3 | RIFE interpolasiyası (→1000 FPS, 2^N + resample) | ✅ hazır |
| Step 4 | Real-ESRGAN 8K upscale (tiled, async I/O) | ✅ hazır |
| Step 5 | FFmpeg yığma + audio mux + verifikasiya | ✅ hazır |
| Step 6 | Resurs təmizliyi və müvəqqəti fayllar | ✅ hazır |
| Step 7 | Checkpoint & resume (çökmədən bərpa) | ✅ hazır |
| Step 8 | CLI qısayolları + Gradio WebUI | ✅ hazır |
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

## Step 3 — RIFE İnterpolasiyası (→1000 FPS)

Step 2-nin 720p kadrlarını rəsmi **RIFE v4** şəbəkəsi ilə 1000 FPS-ə çatdırır:

1. **Model** — rəsmi ECCV2022-RIFE kodu repo-da vendor olunub
   (`pipeline/rife/vendor/`, MIT, yalnız import sətirləri patch olunub);
   çəkilər ilk işə düşmədə Google Drive-dan avtomatik endirilir
   (`weights/rife/`, `--rife-version 4|4.6|hd`, əl ilə `--weights fayl.pkl`).
   İnferens `eval()` + `torch.no_grad()` + CUDA fp16 ilə, 32px padding
   (rəsmi formula) və OOM halında batch-yarıya-bölmə ilə işləyir.
2. **Multi-pass** — `N = ceil(log2(faktor))` rekursiv midpoint-subdivision
   (30→1000 üçün 33.33x → `N=6` → 64x dense), hər səviyyənin midpoint-ləri
   `batch_size` ilə GPU-ya göndərilir.
3. **Dəqiq 1000 FPS** — dense axın (`(P-1)·2^N+1`) axınlı (streaming)
   resample ilə tam `round(müddət·1000)` kadra endirilir — yaddaşda yalnız
   bir cüt saxlanılır.
4. **Sürət kəsələri** — statik cütlər (inference-siz kopiya) və səhnə
   kəsimləri (ghosting-ə qarşı kopiya) avtomatik tutulur (`--no-shortcuts`
   ilə söndürülür).
5. **Yaddaş** — hər 10 cütdən bir `torch.cuda.empty_cache()`, sonda backend
   `unload()` olunur (VRAM Step 4-ə boş qalır); gedişat `tqdm` ilə
   (yazılan/hədəf) göstərilir.

```bash
# Tam zəncir: Step 1 + 2 + 3 (GPU + torch tələb edir):
python main.py --input video/in.mp4 --output video/out.mp4 --to-step 3

# Yalnız Step 3 (saxlanmış konfiqdən):
python main.py --config workspace/config.json --from-step 3 --to-step 3

# CPU-da / torch-suz smoke test (QEYD: real AI deyil, adi blend!):
python main.py --input video/in.mp4 --output video/out.mp4 --to-step 3 \
  --backend blend --output-format jpg

# Köməkçi seçimlər:
python main.py --to-step 3 ... --batch-size 2 --rife-version 4.6 \
  --weights ./train_log --device cuda --max-exp 6 --fp32 --no-shortcuts
```

Nəticə: `interpolated_720p/frame_%08d.png` (8 rəqəm — 1000 FPS!) + yenilənmiş
`config.json` (`interpolation_exp`, `interpolated_frame_count`).

### Proqramlı istifadə (Step 3)

```python
from pipeline.config import PipelineConfig
from pipeline.step03_interpolate import Step03Interpolate

config = PipelineConfig.load("workspace/config.json")
result = Step03Interpolate(backend="rife", batch_size=4).run(config)

print(result.exp, result.target_count)  # məs: 6 2000
print(result.out_dir)                   # Step 4-ün giriş qovluğu
```

## Step 4 — Real-ESRGAN 8K Upscale

Step 3-ün 720p/1000 FPS kadrlarını **8K UHD (7680×4320)**-ə qaldırır:

1. **Model** — rəsmi RRDBNet x4 arxitekturası vendor olunub
   (`pipeline/esrgan/vendor/`, BasicSR Apache-2.0 + Real-ESRGAN BSD-3);
   çəkilər rəsmi GitHub release-dən avtomatik endirilir
   (`--esrgan-model x4plus` ümumi video, `x4plus-anime` animasiya, əl ilə
   `--esrgan-weights fayl.pth`). İnferens `eval()` + `no_grad()` + CUDA fp16.
2. **Ölçü riyaziyyatı** — x4 neyron upscale (5120×2880) + dəqiq Lanczos resize
   ilə tam hədəf ölçü (rəsmi `outscale` proseduru).
3. **Tiling** — rəsmi halo-kəsmə alqoritmi (`tile_pad=10`), tile ölçüsü
   default olaraq Step 1-in VRAM-dəyərindən (`--upscale-tile`, `0` = söndür);
   OOM halında tile avtomatik yarıya bölünüb təkrar cəhd olunur.
4. **Async I/O** — 8K fayllar fon-yazıcı thread-də yazılır (GPU gözləmir),
   sıra qorunur, yazı xətası ucadan bildirilir.
5. **Yaddaş + progress** — hər 10 kadrdan bir `empty_cache()`, sonda `unload()`
   (VRAM Step 5-ə boşalır); `tqdm` ilə kadr/san göstərilir.

```bash
# Tam zəncir 1→4 (GPU + torch tələb edir):
python main.py --input video/in.mp4 --output video/out.mp4 --to-step 4

# Yalnız Step 4 (saxlanmış konfiqdən):
python main.py --config workspace/config.json --from-step 4 --to-step 4

# CPU-da / torch-suz smoke test (QEYD: real AI deyil, adi Lanczos!):
python main.py --input video/in.mp4 --output video/out.mp4 --to-step 4 \
  --backend blend --upscale-backend resize --output-format jpg --upscale-format jpg

# Köməkçi seçimlər:
python main.py --to-step 4 ... --esrgan-model x4plus-anime \
  --upscale-tile 512 --tile-pad 16 --writer-queue 4 --upscale-cache-every 5
```

Nəticə: `upscaled_8k/frame_8k_%08d.png` + yenilənmiş `config.json`
(`esrgan_model`, `upscaled_frame_count`, ...).

### Proqramlı istifadə (Step 4)

```python
from pipeline.config import PipelineConfig
from pipeline.step04_upscale import Step04Upscale

config = PipelineConfig.load("workspace/config.json")
result = Step04Upscale(backend="esrgan", model="x4plus").run(config)

print(result.written_count, result.target_size)  # məs: 2000 (7680, 4320)
print(result.out_dir)                            # Step 5-in giriş qovluğu
```

## Step 5 — Video Yığma + Audio + Verifikasiya

Step 4-ün 8K kadrları və Step 2-nin audiosu tək final videoya yığılır:

1. **Dinamik komanda** — `frame_8k_%08d.*` girişi, dəqiq `-framerate 1000` +
   `-r 1000`, audio varsa `-c:a copy` + dəqiq `-t` trim (səs yoxdursa xətasız
   keçilir), MP4 üçün `+faststart`.
2. **Kodek** — default HEVC: CUDA-da `hevc_nvenc -preset p6 -tune hq -rc vbr
   -cq 19`, CPU-da `libx265 -crf 19 -preset medium`; AV1 seçimləri
   (`av1_nvenc`, `libsvtav1`) də var. Həmişə `yuv420p`. Enkoder FFmpeg
   build-də yoxlanılır, `auto` rejimdə fallback işləyir.
3. **Streamed encode** — kadrlar diskdən axınla gəlir (RAM dolmur),
   `-progress` çıxışı `tqdm` bar-a bağlanır.
4. **Verifikasiya** — çıxış FFprobe (yoxdursa OpenCV) ilə yoxlanılır:
   ölçü/FPS uyğunsuzluğu **xəta** verir, say/davamiyyət fərqi xəbərdarlıq.

```bash
# Tam zəncir 1→5:
python main.py --input video/in.mp4 --output video/final_8k_1000fps.mp4 --to-step 5

# Yalnız Step 5 (saxlanmış konfiqdən):
python main.py --config workspace/config.json --from-step 5 --to-step 5

# GPU enkoder + fərqli keyfiyyət:
python main.py --to-step 5 ... --video-codec hevc_nvenc --crf 17

# Köməkçi: --encoder-preset ultrafast, --ffmpeg-args "...", --skip-verify
```

> **RAM qeydi:** proqram 8K HEVC (libx265) üçün ≥8 GB sistem RAM-ı tövsiyə
> edir — az RAM-da Step 5 əvvəlcədən xəbərdarlıq edir. Çarələr: GPU enkoderi
> (`--video-codec hevc_nvenc`), `.mkv` çıxışı və ya lean x265:
> `--ffmpeg-args "-x265-params pools=1:frame-threads=1:rc-lookahead=10"`.

### Proqramlı istifadə (Step 5)

```python
from pipeline.config import PipelineConfig
from pipeline.step05_assemble import Step05Assemble

config = PipelineConfig.load("workspace/config.json")
result = Step05Assemble(video_codec="auto", crf=19).run(config)

print(result.output_path, result.verified)  # final video + True
```

## Step 6 — Təhlükəsiz Təmizlik (müvəqqəti kadrlar)

Step 5 final `final_8k_1000fps.mp4`-i yaratdıqdan sonra yüzlərlə GB
müvəqqəti məlumat təhlükəsiz silinir:

1. **Təhlükəsizlik qapısı** — final video mövcud və **boş deyilsə** davam
   edir; yoxdursa `CleanupSafetyError` ilə **imtina** edir və heç nə silmir.
2. **Silmə** — `temp_raw_frames`, `interpolated_720p`, `upscaled_8k` və
   çıxarılmış WAV `shutil.rmtree` ilə silinir (`onexc` + köhnə Python üçün
   `onerror` fallback); azad olunan yer loglanır:
   `Cleaned up 142.5 GB of temporary frame data.`
3. **Workspace həbsi** — yalnız `workspace_root` **içindəki** yollar
   silinə bilər (kənardakı audio faylı həmişə saxlanılır).
4. **Xəta tolerantlığı** — kilidli/uğursuz fayllar toplanıb xəbərdarlıq
   verilir, icra yarımçıq qalmır; `--dry-run` silmədən önizləyir;
   `--keep-raw`, `--keep-interpolated`, `--keep-upscaled`, `--keep-audio`
   kateqoriyaları qoruyur.
5. **Step 7-ə hazırlıq** — final video, `config.json` və loglar saxlanılır;
   `cleanup_completed`/`cleanup_freed_bytes` konfiqə yazılır.

```bash
# Yalnız Step 6 (saxlanmış konfiqdən):
python main.py --config workspace/config.json --from-step 6 --to-step 6

# Əvvəlcə önizlə, sonra sil:
python main.py --config workspace/config.json --from-step 6 --to-step 6 --dry-run

# 8K kadrları saxla, qalanını sil:
python main.py --config workspace/config.json --from-step 6 --to-step 6 --keep-upscaled
```

### Proqramlı istifadə (Step 6)

```python
from pipeline.config import PipelineConfig
from pipeline.step06_cleanup import Step06Cleanup

config = PipelineConfig.load("workspace/config.json")
result = Step06Cleanup(dry_run=False).run(config)

print(result.freed_gb, result.failed)  # məs: 142.5 []
```

## Step 7 — Checkpoint & Resume (çökmədən bərpa)

Saatlarla çəkən 8K/1000 FPS emalı elektrik kəsilməsi, VRAM dolması və ya
təsadüfi bağlanma zamanı sıfırdan başlamır — `workspace/pipeline_state.json`
hər addımı izləyir:

1. **Vəziyyət faylı** — son tamamlanmış addım (`last_completed_step`),
   kadr indeksi (`last_processed_frame_index`), giriş video parametrləri və
   Step 2-nin analiz nəticələri atomik yazılır (yarımçıq fayl oxunmur);
   xarab fayl karantinə alınıb təzədən başlanılır.
2. **Avtomatik bərpa sualı** — başlanğıcda yarımçıq iş tapılarsa istifadəçiyə
   iki seçim çıxır: **1)** qaldığı kadrdan davam et (Resume), **2)** tərəqqini
   sil və sıfırdan başla (Overwrite). `--resume yes/no` ilə avtomatlaşdırma
   olur; qeyri-interaktiv shell-də avtomatik resume seçilir.
3. **Kadr-səviyyəli resume** — disk həqiqətdir: yazılmış `frame_*.png`
   faylları skan olunur, AI model dəqiq qaldığı kadrdan/cütdən başlayır
   (Step 3 cüt-səviyyəli, Step 4 kadr-səviyyəli; son fayl şübhəli sayılıb
   yenidən yazılır). Tamamlanmış addımlar (`--to-step` daxilində) keçilir.
4. **OOM bərpası** — `CUDA out of memory` tutulub proqram çökdürülmür:
   checkpoint dərhal yazılır, `tile_size`/`batch_size` yarıya endirilir
   (məs: 512 → 256), GPU keşi təmizlənir və **eyni kadr** təkrar emal olunur.

```bash
# Yarımçıq run-u davam etdir (sual verir):
python main.py --input video/in.mp4 --output video/out.mp4 --to-step 4

# Sualsız resume / sıfırdan başla:
python main.py --input video/in.mp4 --output video/out.mp4 --to-step 4 --resume yes
python main.py --config workspace/config.json --from-step 3 --to-step 4 --resume no

# Checkpoint tezliyi / tam söndürmə:
python main.py ... --checkpoint-every 25
python main.py ... --no-checkpoint
```

### Proqramlı istifadə (Step 7)

```python
from pipeline.checkpoint import CheckpointManager

manager = CheckpointManager("workspace")
decision = manager.decide(manager.snapshot_from_config(config), policy="ask")
print(decision.action)  # "resume" | "overwrite" | "fresh"

# Step 3/4-ə ötür: kadr cədvəli ilə tərəqqi yazılır, OOM-da retry olunur
result = Step04Upscale(checkpoint=manager, checkpoint_every=10).run(config)
```

## Step 8 — CLI Qısayolları + Gradio WebUI

Terminal mütəxəssisləri üçün qısa flag-lər, sıravi istifadəçilər üçün kliklə
işləyən vizual interfeys — hər ikisi eyni `pipeline/webui.py` arxa ucunu
paylaşır (Gradio asılılığı yoxdur, hamısı test olunur):

1. **CLI** — `--resolution 8K|4K|1080p|720p|WxH` (dəqiq
   `--target-width/--target-height` üstün gəlir), `--model` (`--esrgan-model`
   alias-ı; Real-CUGAN planlaşdırılır), `--tile-size auto|256|512|1024`,
   `--ui` (WebUI-ni açır), `--ui-port/--ui-share`.
2. **WebUI (`python app.py`)** — drag & drop video yükləmə, FPS slayderi
   (30–1000), rezolusiya/model/kodek seçimləri, canlı tərəqqi zolağı +
   ETA + GPU temperatur/VRAM, canlı konsol, bitdikdə **Əvvəl/Sonra** video
   müqayisəsi və ekvivalent CLI əmrinin surəti.
3. **Asinxron arxa uc** — emal fon-thread-də gedir (interfeys donmur);
   `ProgressBus` logları, addım bannerlərini və `FrameWatcher` ilə kadr
   sayını brauzerə ötürür; ⏹ cari addımdan sonra dayandırır (Step 7
   checkpoint-i qalır — sonra davam etmək olur).

```bash
# Qısa CLI:
python main.py --input video/in.mp4 --output video/out.mp4 \
  --resolution 4K --model x4plus-anime --tile-size auto --to-step 5

# WebUI:
pip install -r requirements-ui.txt
python app.py                 # və ya: python main.py --ui [--ui-port 7860]
```

### Proqramlı istifadə (Step 8)

```python
import threading
from pipeline.webui import ProgressBus, RunOptions, run_pipeline

opts = RunOptions(input="video/in.mp4", resolution=(3840, 2160))
assert opts.validate() == []
bus, stop = ProgressBus(), threading.Event()
threading.Thread(target=run_pipeline, args=(opts, bus, stop)).start()
# ... bus.poll() ilə LogEvent/StepEvent/FrameProgress oxu, stop.set() ilə saxla
print(" ".join(opts.to_cli_args()))  # ekvivalent CLI əmri
```

### Layihə strukturu

```text
100fps/
├── main.py                      # CLI giriş nöqtəsi (--from-step/--to-step)
├── app.py                       # STEP 8: Gradio WebUI (python app.py / --ui)
├── requirements.txt             # Python asılılıqları
├── requirements-ui.txt          # WebUI üçün əlavə (gradio)
├── pipeline/
│   ├── __init__.py
│   ├── base.py                  # PipelineStep abstrakt bazası (bütün 10 addım üçün)
│   ├── config.py                # PipelineConfig — mərkəzi konfiqurasiya obyekti
│   ├── logger.py                # Vaxt damğalı terminal logları
│   ├── exceptions.py            # Xüsusi xəta tipləri
│   ├── step01_environment.py    # STEP 1: mühit + GPU + konfiqurasiya
│   ├── step02_frames.py         # STEP 2: analiz + audio + kadr çıxarılması
│   ├── step03_interpolate.py    # STEP 3: 2^N subdivision + resample orkestri
│   ├── step04_upscale.py        # STEP 4: 8K upscale orkestri (async yazı ilə)
│   ├── step05_assemble.py       # STEP 5: FFmpeg yığma + audio + verifikasiya
│   ├── step06_cleanup.py        # STEP 6: təhlükəsiz müvəqqəti-məlumat təmizliyi
│   ├── checkpoint.py            # STEP 7: pipeline_state.json + resume + OOM bərpası
│   ├── webui.py                 # STEP 8: RunOptions + ProgressBus + fon-runner
│   ├── frame_io.py              # paylaşılan kadr oxuma/yazma (OpenCV)
│   ├── rife/
│   │   ├── backends.py          # rife (PyTorch AI) / blend (smoke) backend-lər
│   │   ├── weights.py           # çəki həlli + Drive auto-yükləmə
│   │   ├── io.py                # geriyə-uyğun shim (frame_io-ya)
│   │   └── vendor/              # rəsmi RIFE v4 kodu (MIT) + VENDOR.md
│   └── esrgan/
│       ├── backends.py          # esrgan (PyTorch AI) / resize (smoke)
│       ├── tiling.py            # torch-suz tile həndəsəsi (rəsmi riyaziyyat)
│       ├── weights.py           # çəki həlli + GitHub auto-yükləmə
│       ├── writer.py            # fon-thread async yazıcı
│       └── vendor/              # rəsmi RRDBNet (Apache-2.0/BSD-3) + sənəd
└── tests/
    ├── test_step01_environment.py
    ├── test_step02_frames.py
    ├── test_step03_interpolate.py
    ├── test_step04_upscale.py
    ├── test_step05_assemble.py
    ├── test_step06_cleanup.py
    ├── test_step07_checkpoint.py
    └── test_step08_webui.py
```

### Testlər

```bash
pip install pytest
pytest tests/ -v
```
