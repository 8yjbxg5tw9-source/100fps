100fps — 720p videonu 8K @ 1000 FPS-ə çevir (portable buraxılış)
===============================================================

Bu qovluq TAM proqramdır: Python və ya əlavə kitabxana quraşdırmaq lazım
deyil. Qovluğu istənilən yerə köçürüb işlədin (yazma icazəsi olan yer olsun).

TEZ BAŞLANĞIC (GUI — tövsiyə olunur)
  Windows : 100fps-gui.exe-ə iki klikləyin → brauzer avtomatik açılır.
  Linux   : ./100fps-gui  (və ya 100fps-ə arqumentsiz iki klik)
  Sonra videonu sürükləyib buraxın, "Emalı başlat"-a basın.

KOMANDA SƏTRİ (CLI)
  100fps --input video/in.mp4 --output video/out.mp4 --to-step 5
  100fps --help        bütün seçimlər
  100fps --version     versiya

İLK İŞƏDÜŞMƏ
  AI çəkiləri (~130 MB: RIFE + Real-ESRGAN) ilk dəfə lazım olanda rəsmi
  mənbələrdən avtomatik endirilir (tərəqqi zolağı ilə) və models/ qovluğuna
  yazılır. İnternet olmayan maşın üçün "weights" buraxılışını endirin.

QOVLUQ QURULUŞU
  100fps[-gui](.exe)   proqram (CLI + GUI)
  models/              AI çəkiləri (ilk işədüşmədə yaranır)
  bin/                 daxili ffmpeg/ffprobe (silməyin)
  workspace/           emal zamanı yaranır (kadrlar + config + benchmark)
  THIRD_PARTY_LICENSES.txt   üçüncü-tərəf komponentlərin siyahısı

APARAT TƏLƏBLƏRİ (qısa)
  Minimal : 8 nüvə CPU, 16 GB RAM, 50 GB boş disk (CPU rejimi ÇOX yavaşdır)
  Tövsiyə : NVIDIA RTX (12+ GB VRAM), 32+ GB RAM, 200+ GB NVMe SSD
  8K PNG kadrlar nəhəngdir — uzun kliplər yüzlərlə GB istəyə bilər.

PROBLEM HƏLLİ (qısa)
  - "NVIDIA driver tapılmadı" → Studio/Game Ready drayveri yeniləyin.
  - "CUDA out of memory" → proqram tile-i avtomatik kiçildir; alınmasa
    --tile-size 256 və ya --upscale-tile 256 ilə təkrarlayın.
  - Antivirus .exe-ni saxlayırsa → qovluğu istisnaya əlavə edin.
  - Ətraflı: README.md (Tam sənəd) → "Problem həlli" bölməsi.

LİSENZİYA QEYDİ: bu buraxılışdakı FFmpeg GPL/LGPLlidir; AI çəkiləri və
üçüncü-tərəf kitabxanalar üçün THIRD_PARTY_LICENSES.txt-ə baxın.
