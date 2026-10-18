# BitMark to LoRA: radioactivity experiment

Run from the repository root:

```bash
bash run_bitmark_lora.sh
```

The launcher calls an existing Python interpreter with the supplied arguments.
It prefers `/content/bitmark_runtime/.venv/bin/python` (or the runtime root set by
`BITMARK_RUNTIME_ROOT`), then `bitmark_lora/.venv/bin/python`, then `python3`.
It does not install packages, invoke pip/uv, create a venv, or download Python.
Set `BITMARK_PYTHON=/path/to/python` to use another existing interpreter.
Dependencies must already be available in that interpreter. Missing dependencies
produce an import error; the launcher never installs them automatically.

The experiment still downloads model checkpoints/data as needed, generates images,
trains students, runs the detector, and computes CLIP/FID/KID.

## Google Colab

From the repository directory:

```python
!git pull --ff-only
!bash run_bitmark_lora.sh
```

If the Colab system Python cannot import T5, check the existing environment:

```python
!python3 bitmark_lora/check_runtime.py
```

The default command automatically reuses the existing Python 3.11 environment
at `/content/bitmark_runtime/.venv/bin/python` when present. No extra environment
variable is needed. The selected interpreter is printed before launch.

The pipeline checks T5/CLIP/SD/LoRA imports before model downloads or generation.
Backend flags select PyTorch and disable optional TensorFlow/Flax imports. The
diagnostic command shows the original import exception; it does not repair or
modify installed packages.

Source files can reside on Drive, including paths containing spaces. **Runs now
persist beside the root launcher**, independent of the runtime/cache location:

```text
/content/drive/MyDrive/Malicious_quantization/
  run_bitmark_lora.sh
  bitmark_lora/
  run_bitmark_YYYYMMDD_HHMMSS_microseconds/
    run.log
    logs/
    students/
    images/
    result/
```

Launch the copy of `run_bitmark_lora.sh` inside your mounted Drive repository.
The default uses the script's repository directory, not the shell working directory.
`--output /absolute/path/to/new_run` or `BITMARK_RESULTS_ROOT` overrides this.
Only environments, model caches and same-host locks stay under
`/content/bitmark_runtime`. A mounted Drive must remain available while running.
Locks protect simultaneous writers on one host, not different Colab machines.
The GPU must be Ampere or newer (e.g. A100/L4). Preflight checks imports/kernels
and requires 70 GiB free on cache and output volumes by default. The launcher
does not install packages or change the active environment.

## Đánh giá ý tưởng và phạm vi kết luận

Thiết kế đủ để làm pilot kiểm tra **radioactivity phụ thuộc cách adaptation**.
BitMark đã báo cáo truyền watermark sang SD2.1 full fine-tune, nên full là positive
control thích hợp cho việc so với LoRA trên cùng dữ liệu. Mặc định rank 16/32, một seed
và một ngân sách chưa đủ để khẳng định LoRA là điểm yếu chung hoặc đủ novelty cho
một bài top conference. SERUM đã báo cáo radioactivity qua LoRA; không nên đặt giả
thuyết tuyệt đối rằng LoRA không truyền watermark.

Kết quả thuyết phục cần: full truyền rõ, LoRA truyền yếu nhưng vẫn học được phân
phối/capability của teacher, hiệu ứng lặp lại qua rank/LR/ngân sách và seed. Cùng
LR và số step giúp kiểm soát recipe nhưng không bảo đảm hai phương pháp hội tụ
tương đương. CLIP/FID/KID chỉ là kiểm tra utility; ảnh đẹp từ model gốc vẫn có thể
che việc LoRA chưa học đủ. Pipeline đánh giá adaptation trên validation độc lập và lưu đường cong theo ngân sách.
SERUM seed-0 nội bộ đã chạy rank 4 tới 20k nhưng chỉ đạt 4,42% detection;
không có bảo đảm rằng tăng bước hoặc rank sẽ cải thiện radioactivity.

`--train-clean-controls` bổ sung clean-trained LoRA/full để đo interaction
`(full_wm-full_clean) - (lora_wm-lora_clean)` và loại bớt confound từ fine-tuning
nói chung. Rank sweep đã có; có thể thêm full+EMA gần recipe tác giả bằng
`--paper-full-ema`. Những nhánh này tốn thêm compute và chỉ chạy khi bật cờ.
`--seed` hiện thay cả chia prompt, generation và training; đó là lặp toàn pipeline,
chưa phải phép đo chỉ biến thiên seed optimizer trên một dataset cố định.

Nguồn đối chiếu: [BitMark recipe chính thức](https://github.com/sprintml/BitMark#radioactivity)
và [SERUM, Appendix I](https://adam-dziedzic.com/static/assets/papers/serum.pdf).

## Câu hỏi thực nghiệm và đối chứng

Mục tiêu: với dữ liệu BitMark, LoRA có truyền tín hiệu watermark yếu hơn full
fine-tune hay không? Đây là **thí nghiệm truyền watermark qua adaptation**,
không phải quantization attack, không phải thêm watermark trực tiếp vào SD2.1.

Mặc định chạy **4 LoRA + 1 full UNet**, cùng unadapted baseline; không train C00/C01:

| Nhánh | Adaptation | Dữ liệu train |
|---|---|---|
| `unadapted` | Không | Không |
| `lora_r{16,32}_wm` | Attention q/k/v/out | Infinity BitMark |
| `lora_r{16,32}_conv_wm` | Attention + tất cả UNet Conv2d groups=1 | Cùng dữ liệu |
| `full_wm` | Toàn UNet, VAE/text encoder frozen | Cùng dữ liệu |

Cùng SD2.1, 1.000 ảnh train, **7.500 optimizer steps mỗi student**, batch 4,
LR 1e-4, AdamW, clipping 1.0, FP16, resolution 512, không EMA/random flip.
Đây là cùng step budget, **không cùng số tham số hoặc FLOPs**; xem
`students/<branch>/adapter_audit.json` để biết targets và số tham số thực tế.
Chạy tuần tự nên thời gian mặc định dài hơn bản chỉ train rank 4.

### Validation và checkpoint selection

- Prompt train/calibration/validation/test rời nhau cả image ID lẫn caption.
- Validation mặc định 256 ảnh teacher watermark; chọn 128 prompt đầu để sinh ảnh
  đo detection, CLIP/FID/KID. Clean teacher validation làm reference chất lượng.
- Mỗi ảnh validation có một VAE posterior sample cố định. Mỗi checkpoint dùng
  cùng hai noise/timestep draws mỗi ảnh; MSE FP32 hỗ trợ cả epsilon/v-prediction.
- Export snapshots ở **2.500, 5.000, 7.500** bước. Mốc cuối luôn được thêm nếu đổi budget.
  Chỉ lưu inference weights, không nhân bản optimizer state cho mỗi mốc.
- Sau khi train xong một branch, lần lượt reload snapshots để đo MSE, sinh ảnh
  validation, chạy detector chính thức và đo chất lượng. Đây là post-training
  milestone evaluation, không chạy detector trong optimizer loop.
- **Quy tắc chốt trước: MSE validation nhỏ nhất; hòa thì lấy bước sớm hơn.**
  Detection và utility chỉ là diagnostics; chúng không chọn checkpoint.
- Quyết định được freeze trong `result/selection/<branch>.json`. Chỉ sau khi mọi
  branch đã chốt mới sinh test của các student được train; test mặc định 1.000
  prompt riêng. Không chọn nhánh/rank tốt nhất dựa trên test rồi gọi là đánh giá độc lập.
- FID trên 128 ảnh validation là chỉ báo nhiễu và thiên lệch mạnh, không phải
  con số chất lượng đủ để kết luận paper. Dùng test lớn hơn cho kết luận cuối.

`--max-train-steps`, `--checkpoint-steps`, `--validation-n`,
`--validation-generation-n`, `--validation-draws` cho phép thay giao thức trước khi chạy.
`--epochs` vẫn được hỗ trợ như cách tính budget thay thế; không dùng đồng thời
với `--max-train-steps`. Ngân sách mới không được áp lên `--resume` run cũ.

Calibration=5.000 clean teacher; chỉ train trên ảnh watermarked trừ khi bật
`--train-clean-controls`. Teacher Infinity-2B sinh 1024²; trainer resize/crop
512²; student sinh 512²; detector resize theo code chính thức của Infinity.
Clean/watermarked teacher dùng cùng prompt/seed khi có cặp, chỉ khác delta.

### Cải thiện radioactivity: thí nghiệm có căn cứ, chưa phải bảo đảm

[SERUM Appendix I](https://arxiv.org/html/2603.13396) dùng unseen denoising MSE
để chọn checkpoint; pipeline này áp dụng nguyên tắc đó. Kết quả nội bộ không tái
lập được LoRA 52,3%, nên tăng budget được xem là phép kiểm tra, không phải lời giải chắc chắn.
[HMark Table 11](https://arxiv.org/html/2512.00094) báo ảnh hưởng của rank lên
radioactivity trong DDPM; không thể chuyển nguyên số liệu đó sang SD2.1/BitMark.
[PEFT LoRA](https://huggingface.co/docs/peft/v0.17.0/developer_guides/lora)
hỗ trợ convolution adapters. So sánh attention/convolution ở đây kiểm tra giả
thuyết về vị trí học tín hiệu; không thêm watermark loss trực tiếp cho student.

[BitMark chính thức](https://github.com/sprintml/BitMark#radioactivity) dùng
`delta=2` trên mọi scale. Giữ đó làm baseline, thêm tùy chọn
`--watermark-delta 2.5` hoặc `3` cho **run riêng**, cùng seed/budget/protocol.
Cần xem cả quality của teacher lẫn student, không chỉ tăng detector score.
Không tự tuyên bố delta lớn hơn sẽ truyền tốt hơn và không dùng test để chọn delta.

## Nguồn chính thức, không tự viết lại BitMark

- Paper: https://arxiv.org/abs/2506.21209, §4.3 và Appendix F.
- Repo: https://github.com/sprintml/BitMark,
  commit `dd21db32dc86ac70526229e678a027cc06f5c296`.
- Code generator/tokenizer/detector nằm nguyên bản ở `vendor/bitmark/`.
  `teacher.py` gọi `WatermarkInference`, `Infinity.gen_img`, `get_detector`, `detect`.
- `G={01,10}`, context=2, all scales (`watermark_scales=2`), δ=2, CFG=4.
- Infinity/tokenizer: `FoundationVision/Infinity`, revision
  `e7108923249fb3c836ebfbc3f4b4d318d6131293`:
  - `infinity_2b_reg.pth`, SHA256
    `d5958b37e7ecb914230cd2b8c28aae0ec108ce08b315720bc3dc52596cc5b3a7`.
  - `infinity_vae_d32reg.pth`, SHA256
    `7a37fa3ea1b2a1ebd23de61d91a5e68202825e5a67edaef4b7c55f5fd5b9cf26`.
- T5 `google/flan-t5-xl`, revision `7d6315df2c2fb742f0f5b556879d730926ca9001`.
- Student `Manojb/stable-diffusion-2-1-base`, revision
  `0094d483a120f3f33dafbd187ea4aa60d10de75c`: mirror **được README BitMark chỉ rõ**
  vì link StabilityAI cũ không còn khả dụng. Ghi đúng tên mirror trong provenance;
  không tự khẳng định đã đối chiếu byte-for-byte với checkpoint đã bị gỡ.
- Kiểm tra LFS SHA256 của các model đã tải trước khi load; hai file Infinity
  còn có hash hardcode độc lập với metadata Hub. JSON provenance lưu từng file.
- Trainer chính thức Diffusers v0.35.1, commit
  `0f252be0ed42006c125ef4429156cb13ae6c1d60`, được vendor để không bị thay đổi API.
  Vá `lora_layers=filter(...)` thành `list(filter(...))` như fix trong SERUM:
  tránh optimizer dùng cạn iterator khiến clipping mất tác dụng.
  Đồng thời map `--report_to none` thành Python `None` vì Accelerate không chấp
  nhận chuỗi `"none"`; không bật tracker bên ngoài.
  Snapshots LoRA dùng PEFT-native `adapter_model.safetensors` và `adapter_config.json`;
  loader kiểm tra keys/shapes trước khi nạp cả attention và convolution.
  Không dùng file legacy `pytorch_lora_weights.safetensors` để tự reload mixed Conv2d adapters.
  `vendor/provenance.json` lưu hash trước/sau và kiểm tra toàn bộ source khi chạy.

BitMark áp bias **lúc sampling** trên pretrained Infinity; không cần bịa ra một
“watermarked Infinity checkpoint” riêng. Repo tác giả chỉ rõ pretrained weights
và recipe train student, không cung cấp link checkpoint SD2.1 student radioactivity
trong hướng dẫn đó. Pipeline tự train student từ đúng dữ liệu BitMark.

Không dùng `comprehensive_infer.py` của upstream vì file ở commit này có lỗi
`get_detector(args, message)` không đúng signature, tham chiếu `message` không tồn
tại, typo `watermarn_gen_image`, và tổng kết `max()` trên timing list rỗng.
Adapter chỉ thay cách điều phối và ép T5 loader đọc snapshot pin (upstream bỏ qua
tham số `t5_path`); không đổi thuật toán watermark, encoder hay detector.

## Đọc kết quả

```text
run_bitmark_YYYYMMDD_HHMMSS_microseconds/
  config.json, code_manifest.json, environment.json, assets.json
  prompts.json
  run.log                # terminal orchestration và tracebacks
  validation_cache/      # latent/text tensors cố định
  validation/            # MSE theo ảnh tại từng mốc
  logs/                  # môi trường GPU/stage output; lỗi giữ nguyên traceback
  students/              # LoRA/full checkpoint + protocol + verification
  images/                # teacher/student PNG và manifests
  scores/                # signed z, green fraction, SHA256 từng ảnh
  result/
    README.md            # bảng tổng hợp dễ đọc
    checkpoint_metrics.csv # MSE, detection CI, CLIP/FID/KID từng mốc
    selection/           # checkpoint đã chốt theo MSE
    validation_utility.json
    threshold.json       # frozen từ clean calibration trước training
    radioactivity.json   # đầy đủ số liệu, paired CI và cảnh báo
    detection_absolute_z.csv
    detection_signed_z.csv
    utility.json         # CLIP, FID, KID; reference=clean teacher test
```

Paper mô tả detection hai phía `|z|`, trong khi ROC script chính thức dùng signed
`z`. Pipeline **báo cáo cả hai**, lấy `|z|` làm primary được chốt từ trước.
Ngưỡng dùng empirical FPR≤1% trên clean calibration, xử lý ties bảo thủ; cùng
ngưỡng cho mọi branch. Đây là FPR mục tiêu trên calibration, không phải bảo đảm
FPR của mọi phân phối. Calibration ở đây là Infinity clean, không phải natural
COCO; xem false positives trên `unadapted` và `teacher_clean_test`.

Báo cáo mức tăng detection so với `unadapted` cho từng student, và chênh lệch
LoRA/full theo cùng prompt/seed:

```text
TPR(full_wm) − TPR(lora_r4_wm)
```

Báo cáo Wilson CI cho từng detection rate và paired bootstrap CI cho hiệu ứng.
Mặc định không có clean-trained controls nên không gọi các hiệu ứng này là clean-adjusted;
không tách được hoàn toàn ảnh hưởng fine-tuning nói chung. `--analyze` vẫn đọc được
kết quả run cũ có clean-trained controls theo config đã lưu của chúng.
Nếu full cũng chưa truyền được watermark, **chưa thể kết luận LoRA là điểm yếu**.
Nếu LoRA tạo ảnh kém hoặc không học đủ thì TPR thấp cũng chưa chứng minh tính
chọn lọc. CLIP/FID/KID giúp kiểm tra utility, không chứng minh task adaptation;
cần tiếp tục khảo sát rank/LR/seed và kiểm tra ảnh trước khi đưa ra claim mạnh.
`green_fraction` không phải bit accuracy của một message/key như Stable Signature.

Không stop vì TPR hay utility thấp: ghi warning, giữ toàn bộ kết quả. Chỉ lỗi
thực thi, checkpoint/hash sai, ảnh thiếu, nonfinite hoặc config resume khác mới
dừng. Kết quả detection được ghi **trước** bước utility để vẫn xem được nếu tải
Inception/CLIP lỗi. CI ảnh không bao gồm biến thiên seed huấn luyện.

## Chạy mở rộng và resume

```bash
# Mặc định đã so rank 16/32 và attention/convolution.
# Mở rộng ngân sách, giữ cùng mốc cho mọi student:
bash run_bitmark_lora.sh --max-train-steps 10000 --checkpoint-steps 2500 5000 7500 10000

# Run riêng để kiểm tra cường độ nhúng (không đổi student loss):
bash run_bitmark_lora.sh --watermark-delta 3

# Chỉ attention rank 16:
bash run_bitmark_lora.sh --ranks 16 --adapter-scopes attention --lora-only

# Thêm đối chứng full+EMA gần recipe paper, tách khỏi matched full controls:
bash run_bitmark_lora.sh --paper-full-ema

# Pilot mở rộng để kiểm tra capacity và ảnh hưởng fine-tuning nói chung:
bash run_bitmark_lora.sh --ranks 16 32 64 --train-clean-controls --paper-full-ema

# Chạy seed khác độc lập:
bash run_bitmark_lora.sh --seed 1

# Resume giữ NGUYÊN config/source và checkpoint của run:
bash run_bitmark_lora.sh --resume /absolute/path/to/run_...

# Xem cấu hình mà không chạy; đây là lựa chọn tường minh, không phải mặc định:
bash run_bitmark_lora.sh --plan-only

# Tổng hợp lại từ dữ liệu thật, chỉ cần Python standard library:
bash run_bitmark_lora.sh --analyze /absolute/path/to/run_...

# Bộ kiểm tra integrity/statistics/command wiring CPU:
python3 -m unittest discover -s bitmark_lora -p 'test_*.py' -v

# Trong môi trường đã cài dependencies: test thật 2 step/trainer với model tiny
# ngẫu nhiên tạo tại chỗ, reload checkpoint và inference CPU, không tải model:
python3 bitmark_lora/smoke_cpu.py
```

Partial training chưa có `complete.json` sẽ **train lại branch đó từ đầu**;
không giả vờ tiếp tục optimizer. Teacher/student ảnh hoàn chỉnh được reuse với
manifest; checkpoint đã hoàn tất được hash-verify. Trên cùng một host, một run chỉ có một tiến trình
ghi; assets và setup env cũng được khóa khi nhiều job dùng chung cache.

Các thay đổi so với paper cần ghi khi viết báo cáo: fixed seed/prompt split của
pipeline này, paired δ=0 sampler, clean-generated calibration, không xFormers
(dùng Torch SDPA), matched controls không EMA, có cả absolute/signed scoring.
Không được gọi kết quả đó là tái hiện chính xác TPR=98.9% của paper.


## SERUM diffusion teacher

```bash
bash run_serum.sh
```

Root launcher gọi `bitmark_lora/run_serum.py` và dùng chung trainer, validation,
checkpoint reload, utility và reporter đã kiểm thử. `serum/` giữ nguyên thí nghiệm
cũ để đối chiếu; không chạy các script có đường dẫn `/root/workspace` của run cũ.
Luồng mới không nhập code Infinity hay tải T5/BitMark checkpoints.

| Thành phần | Mặc định mới |
|---|---|
| Teacher | SD2.1-base cache gốc + watermark/detector SERUM epoch 50, alpha 0.5 |
| Student | CompVis SD1.4, pinned revision `133a221b8aa7292a167afc5127cb63fb5005638b` |
| Nhánh train | Attention rank 16/32, attention + Conv2d rank 16/32, DoRA attention+conv rank 32, full UNet |
| Ngân sách | 7.500 optimizer steps/nhánh; snapshots 2.500, 5.000, 7.500 |
| Training | 5.000 ảnh teacher watermark, batch 8, AdamW, LoRA LR 1e-5, full LR 5e-6 |
| Validation MSE | 1.000 ảnh watermark rời train/test, 2 fixed noise/timestep draws/ảnh |
| Validation generation | 128 prompt đầu của validation; detection, CLIP/FID/KID ở từng mốc |
| Calibration | 5.000 prompt rời train/validation/test; cả clean teacher và unadapted student |
| Test cuối | 1.000 prompt riêng; mỗi nhánh dùng checkpoint validation MSE tốt nhất |

Đây là cải tiến thực nghiệm về capacity/scope và chọn checkpoint, **không bảo đảm
TPR tăng**. Attention+convolution tốt hơn trong run BitMark trước chỉ là động lực
kiểm tra giả thuyết tương tự ở SERUM. Full control có LR riêng đã khai báo;
không gọi đây là so sánh bằng FLOPs, parameter count hay LR. Mặc định không train
C00/C01. Có thể bật `--train-clean-controls` cho nghiên cứu tách ảnh hưởng generic
fine-tuning; khi không bật, báo cáo chỉ trừ baseline unadapted.

**Chọn checkpoint và tránh leakage:** chỉ MSE validation chọn checkpoint, hòa thì
lấy bước sớm hơn. Detector và FID/KID chỉ báo cáo. Mọi selection được đóng băng
trước khi sinh ảnh test của student đã train. Chất lượng/TPR thấp sinh cảnh báo,
không dừng thí nghiệm; checkpoint sai hash, dữ liệu hỏng hoặc NaN vẫn phải dừng.
Không đánh đồng SERUM probability score với bit accuracy (detector này không giải mã thông điệp bit).

**Hai ngưỡng đã chốt trước:** `detection_serum_score.csv` dùng clean teacher
calibration, để đối chiếu protocol cũ. `detection_serum_score_student_calibrated.csv`
dùng clean unadapted SD1.4 calibration để kiểm tra domain shift. Cả hai dùng cùng
learned detector/VAE SD2.1, không fit trên student test hay student đã fine-tune.
Báo cáo Wilson CI, paired contrasts và FPR của ảnh sạch. Ngưỡng teacher có thể có
FPR cao trên student; không dùng riêng raw TPR đó làm bằng chứng radioactivity.
Chạy nhiều seed độc lập trước khi kết luận tổng quát; CI theo ảnh không bao hàm
biến thiên của quá trình train.

**Nguồn thật:** [tác giả cung cấp checkpoint và cache SD2.1](https://github.com/Hubizon/SERUM/issues/1#issuecomment-4963328577).
`serum_assets.py` pin SHA256 checkpoint
`e991a694342545fa4f59cc5b3c94b6e3a93bf831dd4d507207f52d1c167c05c4`
và archive
`95877f6ab0305b92897c5f6fdba7fcd373b9031728f3e69599ffebfb6d2bc593`,
theo manifest của run SERUM gốc trong repo. Verify trước `torch.load(weights_only=True)`;
không tự thay bằng watermark ngẫu nhiên hoặc checkpoint mirror.
`vendor/serum` giữ forward Watermark/WatermarkScoreModel chính thức, commit
`02f4929e30cd976c5cfe65a45ef1bc4ac79bdfed`; chỉ chỉnh đường import config/cleanup.
VAE posterior sampling dùng RNG theo image index, chung giữa các nhánh; không
sao chép threshold cũ vì batching/RNG protocol đã được khai báo lại.

Google Drive có thể giới hạn tải hoặc bị chặn trên một số máy. Nếu đã tải artifact
chính thức về, truyền `--serum-checkpoint /path/latest_checkpoint.pt` và
`--serum-cache-archive /path/sd21_original_cache.zip`; SHA256 vẫn được kiểm tra.
Không tắt checksum khi download lỗi. Cache/env ở local runtime, kết quả ở cạnh repo:

```bash
# Chạy mới đầy đủ, kết quả persist cạnh code trên Drive:
bash run_serum.sh
# Tiếp tục đúng cấu hình đã ghi; không ghép các override mới:
bash run_serum.sh --resume /path/to/run_serum_YYYYMMDD_HHMMSS_microseconds
# Chỉ xem kế hoạch (lệnh này không train):
bash run_serum.sh --plan-only
# Dùng Python environment hiện có khác:
SERUM_PYTHON=/path/to/python bash run_serum.sh
```

SERUM dùng lại các dependency ML trong `requirements.txt`, nhưng không cần
FlashAttention. Launcher không cài/gỡ package trong job. Toàn trình mặc định gồm
nhiều lượt sinh ảnh calibration/validation/test và 6 lượt train nên cần GPU và thời
gian đáng kể. Mỗi stage in tiến độ và lưu `logs/`, `run.log` trong thư mục run.
Bộ kiểm tra hiện có: 38 protocol tests; CPU tiny diffusion train/reload/inference
cho attention, attention+conv và full; smoke forward SERUM với **synthetic weights**.
Đây không thay cho chạy checkpoint tác giả và đo radioactivity thật trên GPU.


### Nhánh thử nghiệm DoRA cho SERUM

`bash run_serum.sh` giờ chạy thêm **`dora_r32_conv_wm`** bên cạnh
`lora_r32_conv_wm`. DoRA dùng PEFT `use_dora=True`, học độ lớn trọng số riêng cùng
hướng cập nhật low-rank. Không đổi watermark/detector tác giả, không lấy score
watermark làm loss train. Giả thuyết: thêm khả năng điều chỉnh magnitude có thể
thu hẹp khoảng cách full UNet–LoRA trong việc học tín hiệu SERUM.

Nguồn: [DoRA, ICML 2024](https://proceedings.mlr.press/v235/liu24bn.html),
[code tác giả](https://github.com/NVlabs/DoRA),
[PEFT 0.17 hỗ trợ Linear và Conv2d](https://huggingface.co/docs/peft/v0.17.0/package_reference/lora).
Tác giả cũng ghi rõ DoRA trên diffusion còn mang tính thử nghiệm và có thể cần
hyperparameter khác. Đây là một ablation có căn cứ, **không phải thuật toán mới
hay kết quả radioactivity đã được chứng minh**.

- Mặc định rank 32, cùng attention+conv scope, LR 1e-5, 7.500 bước, dữ liệu,
  prompt/seed và quy tắc chọn validation checkpoint với LoRA tương ứng.
- Nếu đổi `--ranks`, DoRA dùng rank lớn nhất trong danh sách. `--no-dora` tắt
  nhánh này. BitMark vẫn mặc định tắt DoRA; có thể bật bằng `--dora`.
- Thêm parameters magnitude và chi phí tính toán; không tuyên bố bằng FLOPs.
  `adapter_audit.json` ghi method, tổng trainable và `magnitude_parameters`.
- `result/radioactivity.json` có `dora_minus_lora_conv_r32` và paired bootstrap CI
  theo cả hai ngưỡng. So cùng `utility.json`, checkpoint MSE và FPR sạch trước
  khi kết luận DoRA tốt hơn. CI theo ảnh không thay cho nhiều training seed.
- Native PEFT checkpoints giữ cả A/B và magnitude. CPU smoke kiểm tra magnitude
  đổi qua các bước, reload tái tạo dự đoán, rồi sinh ảnh bằng model nhỏ.
- Run cũ không có trường `dora` vẫn giữ nguyên các nhánh cũ khi đọc config;
  thay đổi code không tự thêm DoRA vào run đã đóng băng. Chạy thư mục mới để thử
  mặc định mới; không sửa manifest của run đang chạy.

Lệnh chạy toàn bộ nhánh mới:

```bash
bash run_serum.sh
```


### Khôi phục lỗi full UNet train xong nhưng export pipeline thất bại

Nếu log đã tới 7.500/7.500 rồi báo thiếu `pytorch_model.bin`/`model.safetensors`
ở bước cuối `train_text_to_image.py`, nguyên nhân đã sửa là pipeline export cố
nạp safety checker vốn không có trong cache thí nghiệm. Không xóa thư mục run.

```bash
git pull origin main
bash run_serum.sh --resume /path/to/run_serum_YYYYMMDD_HHMMSS_microseconds --recover-full-export
```

Lệnh recovery kiểm tra protocol/dữ liệu train và **tất cả snapshot đã khai báo**,
kiểm tra keys/shapes/finite so với base UNet và xác nhận trọng số đã đổi. Sau đó
khôi phục export UNet từ snapshot đúng `max_train_steps`, tiếp tục validation,
selection và test; không gọi trainer. Chọn checkpoint vẫn bằng MSE trên toàn bộ
milestones, không tự chọn snapshot cuối làm kết quả test.

Đây là chế độ sửa export riêng: không vượt qua `code_manifest.json` một cách im
lặng hay sửa provenance gốc. Thay đổi source được ghi riêng trong `recovery/`,
khôi phục weights ghi trong `students/full_wm/export_recovery.json`. Nếu nhánh
khác chưa train xong, thiếu snapshot hoặc tensor hỏng, lệnh báo lỗi rõ ràng;
không giả vờ hoàn thành và không tự train lại. Log tiếp tục lưu trong run cũ.
Bộ kiểm tra: 39 tests trong ML environment; tiny diffusion smoke dùng metadata
khai báo safety checker nhưng không tải checkpoint đó để tái hiện tình huống này.
