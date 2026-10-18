# Hướng dẫn chạy thí nghiệm malicious quantization cho watermark diffusion

## SERUM: diffusion teacher → diffusion student (15/10/2026)

```bash
bash run_serum.sh
```

Lệnh này **chạy thí nghiệm thật**, không phải plan-only. Dùng Python/environment
đã chạy BitMark; không cần Infinity/FlashAttention cho nhánh SERUM. Teacher là
SD2.1 + checkpoint watermark/detector SERUM của tác giả, student là SD1.4.
Mặc định bốn LoRA **rank 16/32 × attention/attention+convolution**, thêm full UNet và **DoRA rank 32 attention+convolution** thử nghiệm;
**7.500 bước**, đánh giá snapshots **2.500 / 5.000 / 7.500**, không chạy rank 4
hay mốc 1.250 trong cấu hình mặc định mới.

Logs, ảnh, checkpoints và báo cáo lưu cạnh repo trong
`run_serum_<ngày_giờ>/`; nếu repo ở Drive thì kết quả cũng ở Drive.
Xem `result/README.md`, `result/checkpoint_metrics.csv`, `result/radioactivity.json`
và `result/utility.json`. [Protocol, nguồn checkpoint và resume](bitmark_lora/README.md#serum-diffusion-teacher).

## BitMark → LoRA radioactivity (13/10/2026)

Thí nghiệm riêng dựa trên protocol SERUM, dùng source/checkpoint BitMark chính
thức: so sánh LoRA và full fine-tune SD2.1 trên dữ liệu watermarked, kèm unadapted
baseline. Không chạy các nhánh train clean kiểu C00/C01. Chạy toàn trình bằng Python environment đã chuẩn bị:

```bash
bash run_bitmark_lora.sh
```

Code, protocol, cách resume và các giới hạn diễn giải: [bitmark_lora/README.md](bitmark_lora/README.md).
Mặc định so rank 16/32 × attention hoặc attention+convolution, cùng 7.500 bước.
Đo MSE/detection/CLIP/FID/KID ở 2.500/5.000/7.500; chọn checkpoint bằng validation MSE,
đánh giá cuối trên test riêng. Kết quả và log nằm cạnh code (trên Drive nếu code ở Drive):
`run_bitmark_<ngày_giờ>/result/README.md`; đường cong: `result/checkpoint_metrics.csv`.

## SleeperMark: quality-constrained public-trigger W4 control (01/10/2026)

The separate public-trigger distillation **defense control** matches the
marked FP32 teacher on ordinary and triggered TRAIN conditions while fitting
a W4 UNet student. It runs independently of the attack branch:

```bash
bash run_blind_quantization.sh --watermark sleepermark --methods public_trigger_distill_defense_w4
```

See [why LLM watermark inheritance is only an analogy](survey/LLM_Distillation_Radioactivity_to_SleeperMark_VI.md).

The 30/09/2026 six-branch evolutionary/LLM suite retained TPR 100/100 for every
branch (bit accuracy 96.625–99.3125%). FIT/SELECT proxy gains did not transfer to
owner TEST. A separate public-trigger QAT run reached 0/100 detection and 49.4%
bit accuracy but damaged images (triggered SSIM 0.551, FID 135.7, only 2/100
joint quality/evasion successes). These are distinct threat models.

The experimental `public_trigger_rollout_qat` branch keeps hard W4 UNet weights,
adds truncated two-step ordinary-prompt DDIM preservation, and selects among
intermediate checkpoints by image metrics on prompt-disjoint TRAIN data. It uses
the published trigger; it never uses the owner key, extractor or TEST for training
or selection. Its effectiveness is unverified until the full owner evaluation.

```bash
bash run_blind_quantization.sh --watermark sleepermark --methods public_trigger_rollout_qat
```

See [protocol and limitations](survey/SleeperMark_Quality_Constrained_Rollout_VI.md).

## Thử hyper-heuristic SleeperMark (30/09/2026)

Chạy các đối chứng W4 model-only cùng 12 × 13 = 156 genome đã đánh giá trên FIT,
chọn checkpoint bằng SELECT rồi tự đánh giá owner TEST:

```bash
bash run_blind_quantization.sh --watermark sleepermark --methods adaptive_genetic_w4 behavior_archive_genetic_w4 bandit_hyperheuristic_w4 llm_hyperheuristic_w4
```

`behavior_archive_genetic_w4` giữ hành vi noise-prediction đa dạng trong quality corridor.
`bandit_hyperheuristic_w4` chọn operator bằng FIT reward; `llm_hyperheuristic_w4`
dùng Qwen2.5-3B-Instruct đề xuất **xác suất operator**, bước mutation và mức lai
giữa các hành vi khác nhau mỗi ba thế hệ. LLM không sinh code, không đọc key,
extractor, SELECT hay TEST, và có schema validation/fallback. Kết quả owner chỉ
được đo sau khi SELECT đóng băng checkpoint. Thời gian/token của LLM là chi phí
bổ sung dù số lần đánh giá genome bằng nhau.

Đối chứng **dùng trigger công khai** được báo riêng, không phải model-only:

```bash
bash run_blind_quantization.sh --watermark sleepermark --methods public_trigger_genetic_w4 public_trigger_bandit_w4
```

Các nhánh mới chưa bật mặc định và chưa có kết quả chứng minh giảm watermark.
Xem [thiết kế và quy tắc đánh giá](survey/SleeperMark_Hyperheuristic_LLM_Evolution_VI.md).

Thử nghiệm mới SleeperMark: genetic search trên scale/rounding W4 và random
search cùng ngân sách (chưa bật mặc định vì chưa có kết quả xác nhận):

```bash
bash run_blind_quantization.sh --watermark sleepermark --methods genetic_quantizer_w4 random_quantizer_w4
```

Mặc định mỗi nhánh 156 candidate, không dùng owner key/extractor để chọn.
`--steps` không điều khiển GA; dùng `--evolution-population` và
`--evolution-generations`. Xem [protocol và giới hạn](survey/SleeperMark_Genetic_Search_VI.md).

Ứng dụng survey: crossover theo cặp scale–rounding, chọn theo đa dạng prediction,
operator thích nghi và proxy residual subspace, kèm hai đối chứng:

```bash
bash run_blind_quantization.sh --watermark sleepermark --methods adaptive_genetic_w4 subspace_genetic_w4 random_subspace_genetic_w4 quality_genetic_w4
```

Bốn nhánh này dùng cùng ngân sách 156 candidate/nhánh. `random_subspace_genetic_w4`
dùng basis ngẫu nhiên nhưng vẫn chạy GA cải tiến; khác với random search cũ.
Chưa bật vào mặc định. Xem [kết quả cũ, công thức proxy và protocol mới](survey/SleeperMark_Structured_GA_Residual_Proxy_VI.md).

GA spatial-MSE hiện áp dụng `--spatial-loss-mode noise` (mặc định), chuẩn hóa
loss theo timestep như QAT. Dùng `--spatial-loss-mode x0` để giữ objective GA
cũ; residual-subspace vẫn dùng chuẩn hóa energy từ FIT. `selection_diagnostics`
trong `*_search.json` ghi rõ chọn RTN fallback hay candidate mới và mức giảm
proxy loss. Gate noise loss không bảo đảm chất lượng ảnh hoặc watermark yếu đi.
Xem [đánh giá GA/random đã hoàn tất](survey/Review_SleeperMark_GA_20260927_VI.md).

Thử basis residual cục bộ (giữ global mặc định để đối chiếu):

```bash
bash run_blind_quantization.sh --watermark sleepermark --methods subspace_genetic_w4 random_subspace_genetic_w4 --evolution-subspace-layout patch3
```

Xem [động cơ, số liệu tổng quát hóa và giới hạn của patch proxy](survey/SleeperMark_Patch_Residual_Proxy_VI.md).

Tiến trình chỉ chạy GA nay lập trước FIT/SELECT để chỉ sinh trajectory được
dùng (tối đa 32 thay vì 256 ở mặc định), giữ nguyên seed và record đã chọn.
Manifest có `evolution_calibration_plan` để kiểm tra. QAT vẫn dùng đầy đủ
calibration. Xem [kết quả cuối structured GA](survey/Review_SleeperMark_Structured_Final_VI.md).

Thử LLM đề xuất quantizer, cùng local-search control và adaptive GA:

```bash
bash run_blind_quantization.sh --watermark sleepermark --methods adaptive_genetic_w4 local_proposal_genetic_w4 llm_genetic_w4
```

LLM chạy local, model/revision được pin, chỉ nhận FIT archive và không thực thi
code sinh ra. Ba proposal/thế hệ thay thế offspring GA nên vẫn có 156 candidate;
token, thời gian và VRAM LLM vẫn là compute bổ sung. Xem [thiết kế, survey và
protocol](survey/LLM_Guided_Quantization_Search_VI.md).

Kết quả LLM/local/GA trên SleeperMark đều có TPR 100/100; xem
[báo cáo](survey/Review_SleeperMark_LLM_20260927_VI.md). Thử proxy mới giảm
đáp ứng high-pass với prefix ngẫu nhiên, giữ W4 và cùng protocol FIT/SELECT:

```bash
bash run_blind_quantization.sh --watermark sleepermark --methods prefix_invariance_genetic_w4
```

`--evolution-prefix-probes` mặc định là 4. Prefix được lấy từ TRAIN, độc lập
với trigger/key/extractor của owner; đây là giả thuyết cần kiểm nghiệm, chưa có
owner result chứng minh hiệu quả.

Lần chạy đầu của nhánh prefix chọn lại RTN. Search evolutionary hiện ưu tiên
ứng viên đạt chất lượng trên FIT trước khi tối ưu proxy, rồi vẫn kiểm tra SELECT
độc lập; dùng `--evolution-legacy-search` để chạy lại search cũ. Đối chứng
`public_trigger_genetic_w4` dùng trigger công khai trong paper nhưng vẫn giữ
key/extractor và owner TEST ngoài vòng chọn. Chạy ba nhánh để tách ảnh hưởng
của cách tìm quantizer và thông tin trigger:

```bash
bash run_blind_quantization.sh --watermark sleepermark --methods adaptive_genetic_w4 prefix_invariance_genetic_w4 public_trigger_genetic_w4
```

Xem [lý do, threat model và protocol](survey/SleeperMark_Quality_Corridor_Public_Trigger_VI.md).
Đối chứng gradient `public_trigger_consistency_qat` có cùng side information
và xuất checkpoint W4; mặc định chạy 2.000 bước khi được chọn tường minh.

**Cấu hình hiện hành, 27/09/2026:** `transfer` không chạy bất kỳ nhánh FP32
fine-tune nào, không chạy fine-tune → RTN và không tạo purified teacher ngầm.
Các implementation này vẫn có trong `science`/`full` hoặc lệnh tường minh.

Stable Signature chạy `fixed_ptq`, `reconstruction`, `natural_rounding`,
`natural_residual`, `natural_residual_qat_warm` ở W4, thêm nhánh thử
`natural_delta_residual` với **FP32 base + delta INT4**. Nhánh cuối không phải
whole-model W4 và được báo trong comparison group riêng.
Warm-QAT mặc định dùng `--warm-checkpoint-policy final`: checkpoint cuối ngân sách
2.000 bước, không chọn theo owner TEST. SEARCH vẫn được ghi để phân tích.
Muốn trở lại chính sách cũ: `--profile transfer -- --warm-checkpoint-policy search`.

SleeperMark chạy `fixed_ptq`, `cfg_reconstruction` ở W4
group 64, `delta_cfg_equivariance` với FP32 base + delta INT4 per-channel,
và hai nhánh thử `coherent_probe_qat`
(W4) / `delta_coherent_probe` (FP32 base + delta4).
Các nhánh SleeperMark mặc định chạy bằng các process CUDA độc lập, đồng thời theo
VRAM đang trống. Hiện có năm nhánh mặc định; số chạy cùng lúc phụ thuộc bộ nhớ. Kết quả được gom
vào `output_attack/sleeper_parallel_*/parallel_summary.csv`; log của mỗi nhánh nằm
cùng suite. Concurrency được tính từ VRAM, không hardcode tên H200.

```bash
# Giới hạn rõ số process nếu đang dùng chung GPU
bash run_blind_quantization.sh --watermark sleepermark --parallel-branches 2

# Chế độ tuần tự cũ để debug hoặc GPU ít VRAM
WMQ_SLEEPER_SEQUENTIAL=1 bash run_blind_quantization.sh --watermark sleepermark
```

Mặc định bộ điều phối dự phòng 8% tổng VRAM (ít nhất 8 GiB), ước lượng 18 GiB
mỗi process và chỉ chạy tối đa số nhánh thực có. Có thể chỉnh ước lượng bằng
`--parallel-vram-per-process-gib` và phần dự phòng bằng
`--parallel-reserve-vram-gib`. Đây là cơ chế tránh OOM, không phải cam kết mỗi
process thực sự dùng đúng 18 GiB. Cấu hình mới dành thêm headroom cho probe batch 2.
Chạy song song có thể tăng throughput của cả suite;
nếu một nhánh đơn đã bão hòa compute, thời gian riêng của nó có thể tăng.

Calibration tensors mặc định được cache một lần trên GPU nếu kích thước ước tính,
50% headroom và phần dự phòng 5% VRAM còn vừa. Việc này giảm copy lặp lại mà không
đổi dữ liệu/loss. Dùng `--calibration-cache cpu` để tắt hoặc `cuda` để yêu cầu bắt
buộc và dừng rõ ràng nếu không đủ bộ nhớ.
`adversarial_equivariance_qat`, `equivariance_qat`, `conditional_rollout_qat`
giữ trong code nhưng không chạy mặc định sau khi kết quả không cải thiện attack
ở chất lượng chấp nhận được. Nhánh delta control dùng mục tiêu spatial trên CFG
prediction và giữ nguyên marked UNet làm base; không tải model sạch thay thế.

[Phân tích ý tưởng DeltaZip](survey/Review_Delta_Warm_Sleeper_20260927_VI.md).
Run SleeperMark `20260927_013638_836714`: delta đạt SSIM triggered 0.897 nhưng
TPR vẫn 100%; tất cả nhánh có joint success 0/100.
[Bảng kết quả và cải tiến cả bốn nhánh](survey/Review_SleeperMark_20260927_013638_VI.md).
Mặc định mới dùng `--weight-init mse --quant-refinement balanced`: khởi tạo W4
theo sai số trọng số, tách LR code/scale, lấy mẫu đều các timestep; equivariance
và delta dùng CFG spatial loss quy đổi về noise prediction cùng TRAIN context
augmentation. Run H200 `sleepermark_20260927_061035_583957` đã hoàn tất: W4 học
đưa BA xuống 94–95% nhưng TPR 98%, SSIM triggered 0.62–0.63; delta giữ SSIM
0.959 nhưng TPR 100%. Mọi nhánh joint success 0/100. Đây chưa phải attack hiệu quả.
Để đối chiếu đúng bốn nhánh cấu hình trước:
`bash run_blind_quantization.sh --watermark sleepermark --methods fixed_ptq cfg_reconstruction equivariance_qat delta_cfg_equivariance --weight-init rtn --quant-refinement legacy`.

`conditional_rollout_qat` chỉ sửa spatial defect của phần conditional-minus-unconditional,
nhằm tránh sửa cả sai lệch không gian vốn có của dự đoán vô điều kiện. Nhánh này
học hai bước DDIM liên tiếp trên hai quỹ đạo riêng của teacher đã hiệu chỉnh và
student W4; detach trạng thái giữa các bước để giữ bộ nhớ thấp. Chỉ code/scale
quantizer được học, không dùng key/extractor/trigger hoặc model sạch. Run hoàn tất
`20260927_075826_683759` đạt BA 94.23%, TPR 98%, triggered SSIM 0.632, joint success
0/100, gần như ngang equivariance. Đây không phải full-trajectory backprop.
Xem [báo cáo server và giả thuyết cải tiến](survey/Review_SleeperMark_Server_20260927_VI.md).

Chạy lại nhánh rollout để đối chứng, kèm baseline FP32 tự đánh giá:

```bash
bash run_blind_quantization.sh --watermark sleepermark --methods conditional_rollout_qat
```

Đối chứng cùng conditional target nhưng một bước: thêm `--rollout-horizon 1`.
Mặc định hai bước tăng chi phí huấn luyện riêng nhánh mới; không chạy W8 hoặc
FP32 fine-tune, không đổi thuật toán sinh ảnh lúc inference.

Hai nhánh coherent probe tìm một perturbation nhỏ dùng chung trên embedding của
ít nhất hai prompt TRAIN khác nhau, cùng timestep. Probe tối đa hóa phần response
high-pass lặp lại giữa các ảnh và phạt thay đổi low-pass. Quantizer học giảm thành
phần chung khi pairwise agreement dương; ordinary CFG reconstruction vẫn bảo toàn
hành vi thông thường. Không dùng trigger/key/extractor lúc học; continuous context
không phải trigger đã phục hồi. Đây là giả thuyết mới, chưa có kết quả full-run.
Xem [review kết quả và mục tiêu coherent probe](survey/Review_SleeperMark_Coherent_20260927_VI.md).

```bash
# Chỉ hai nhánh mới, chạy song song và tự đánh giá/FID
bash run_blind_quantization.sh --watermark sleepermark --methods coherent_probe_qat delta_coherent_probe
```

`--coherent-batch-size 2`, `--coherent-weight 1`, `--coherent-semantic-weight 1`
là mặc định. Riêng W4 coherent dùng code LR 0.003, scale LR 0.0001 để hạn chế
drift; delta dùng delta LR như trước và scale LR 0.0001. `--probe-steps 0` tạo đối
chứng shared random context không tìm bằng gradient. Các nhánh mới chưa được chứng
minh hữu ích; giữ đối chứng để tách chất lượng, representation và mục tiêu mới.

**Chạy cả hai watermark bằng một lệnh:**

```bash
bash run_blind_quantization.sh
```

Mặc định chạy tuần tự **Stable Signature (profile `transfer`) → SleeperMark**,
tự chuẩn bị môi trường/checkpoint, đánh giá watermark và tính FID cho cả hai.
Hai tiến trình không dùng GPU đồng thời. Nếu Stable Signature lỗi, launcher dừng
và báo lỗi; SleeperMark chưa được chạy.

**Mới: SleeperMark — can thiệp trực tiếp UNet và tự đánh giá watermark/FID:**

```bash
bash run_blind_quantization.sh --watermark sleepermark
```

Lệnh tự chuẩn bị môi trường, tải UNet SleeperMark chính thức và backbone SD1.4,
chạy 5 phương pháp W4 (kèm đối chứng FP32 cho fine-tune), rồi đánh giá prompt có
trigger và prompt thường. Mặc định seed 3407, 2.000 bước, batch train 1,
256 ảnh calibration mỗi hướng, 100 prompt test. Scope mặc định là các ma trận
attention ở UNet up-blocks; `--scope all` mở rộng sang tất cả ma trận UNet và tốn
thêm bộ nhớ. VAE/text encoder được giữ nguyên và kiểm tra hash.

Thử luồng ngắn trước khi treo máy:

```bash
bash run_blind_quantization.sh --watermark sleepermark --steps 2 --train-n 2 --test-n 2
```

Xem [protocol và giới hạn SleeperMark/FID](survey/SleeperMark_UNet_FID.md).
Các nhánh mới chưa có kết quả attack trên GPU; không suy ra hiệu quả từ việc chạy được.

**FID cho cả Stable Signature:** các lần chạy mới tự ghi `fid.csv`/`fid.json`
sau owner evaluation. Mặc định là **FID tới output model fingerprint ban đầu**,
đo độ lệch phân phối; không phải FID tới ảnh thật. Để thêm FID tới tập ảnh thật
held-out riêng, đặt `WMQ_FID_REAL_REFERENCE=/path/to/real_test_images` cho Stable
Signature, hoặc `--fid-real-reference /path/to/real_test_images` cho SleeperMark.
Không dùng ảnh natural TRAIN làm tập tham chiếu để tuyên bố khả năng tổng quát.
SleeperMark còn ghi `ordinary_fid.csv` cho prompt thường; `fid.csv` là prompt trigger.
JSON/CSV ở `output_attack`, ảnh/checkpoint/cache ở `output_artifacts`.

FID với 100 ảnh chỉ là chẩn đoán, có cờ `small_sample_warning`; không so trực tiếp
với FID 30k của paper. SleeperMark chỉ **báo cáo** chất lượng, không loại checkpoint
vì SSIM. Hướng Stable Signature giữ nguyên quality policy đã cấu hình; thêm FID
không tự ý thay đổi policy hay chọn checkpoint bằng TEST. Có thể tính lại FID
cho output cũ nếu vẫn còn ảnh tại đường dẫn trong manifest:

```bash
conda run -n wmq python src/wmq_fid.py --run output_attack/<run>
```

`WMQ_EVAL_FID=0` tắt FID ở Stable Signature; `--skip-fid` tắt ở SleeperMark.
Nếu FID lỗi tải mạng, kết quả owner vẫn được giữ; xem `fid_error.json` và dùng
lệnh tính lại ở trên. Với ảnh SleeperMark không trigger, thêm `--ordinary`.

**Chạy riêng Stable Signature:** dùng `bash run_blind_quantization.sh --profile transfer`
với đúng một seed `3407`, một preservation weight `0.5`. Profile `science` vẫn có
thể chọn tường minh. Kiểm tra kế hoạch Stable Signature không chạy GPU bằng
`bash run_blind_quantization.sh --profile transfer --plan-only`. Các flag của `wmq_blind.py`
đi sau `--` trong suite, ví dụ `-- --train-batch-size 2`. Lệnh tùy chỉnh ở cấp
thí nghiệm dùng `bash run_blind_quantization.sh --direct --model ...`.
`run_blind_suite.py` vẫn là orchestrator nội bộ, không cần shell launcher thứ hai.

`natural_joint_quality_finetune` dùng activation checkpointing cho bốn decoder
forward và hai encoder-cycle forward trong mỗi update. Batch và số ảnh/step không
đổi, nhưng backward sẽ tính lại activation nên chậm hơn để giảm peak VRAM. Selection
ghi `activation_checkpointing=true` và `peak_cuda_allocated_gib`. Nếu một branch OOM,
launcher loại partial branch, chạy GC + CUDA cache/IPC cleanup rồi vẫn thử held-out
evaluation cho các checkpoint đã đóng băng. Cleanup không thể giải phóng VRAM của
process khác; hãy chạy `nvidia-smi` trước khi khởi động suite.

**Lịch sử run `20260923_171848`, không còn là mặc định:** profile `transfer` từng giữ joint fine-tune cũ và
thêm `natural_joint_quality_finetune` + bản RTN W4. Nhánh mới phạt riêng từng ảnh
generated TRAIN có PSNR dưới `min_image_psnr` (mặc định 25 dB) hoặc SSIM dưới
`min_ssim` (0.80), cho cả FP32/W4. Trọng số `--joint-quality-weight` mặc định 0.01;
SEARCH thêm cùng penalty, không dùng owner TEST. Giữ nguyên quality gate và
tiếp tục ghi kết quả nếu không đạt. Nhánh này vẫn ngoài threat model quantizer-only.
Warm-QAT hiện đã được bật lại theo cấu hình đầu README; các nhánh fine-tune đã tắt.
Xem [review và hướng cải tiến](survey/Review_20260923_171848_VI.md).

**Thử nghiệm fine-tune tùy chọn trong `science`:**
`natural_joint_finetune`: fine-tune decoder bằng cả reconstruction FP32 và W4,
kèm cycle latent qua encoder cố định. Nhánh xuất riêng FP32 và
`natural_joint_finetune_rtn_w4` rồi tự owner-evaluate cùng các đối chứng.
Vẫn 2.000 update, SSIM trung bình ≥0.80. Đây là unrestricted fine-tune + lượng tử hóa,
**không thuộc quantizer-only**; chưa có kết quả chứng minh tốt hơn fine-tune thuần.
Mỗi bước cần thêm lượt decoder/encoder nên chi phí cao hơn. Code chạy tiếp và
ghi nhận nếu không đạt quality gate. Chi tiết và ablation:
[Joint FP32/W4 fine-tuning](survey/Joint_Finetune_VI.md).

Nhánh tùy chọn trong `science`: `natural_residual_cycle_qat_warm`.
Chạy `bash run_blind_quantization.sh --profile science` để so với warm-QAT cùng W4,
2.000 bước và ngưỡng SSIM trung bình 0.80. Nhánh này khởi tạo từ cùng residual W4
đã chọn trên SEARCH, thêm loss encode(decode(z)) khớp latent tự nhiên ban đầu.
Encoder cố định, gradient truyền qua encoder về quantizer; không dùng key/extractor.
Loss được chia cho năng lượng latent từng ảnh (chặn dưới 1e-4), trọng số mặc định
`--cycle-weight 0.01`; đặt 0 để làm ablation. Xem `natural_validation_cycle` trong
search và `cycle_loss` trong updates, cùng số lượt encoder trong selection report.
Đây là giả thuyết bảo toàn nội dung, chưa chứng minh làm yếu watermark hay có novelty.
Nó tăng thời gian/bộ nhớ do thêm encoder backward. So hiệu quả với warm-QAT ở cùng
chất lượng, báo cả chi phí nhánh residual dùng để khởi tạo. Không chọn bằng owner TEST.
Xem phân tích tại [Cycle warm-QAT](survey/Cycle_Warm_QAT_VI.md).

## Lệnh mặc định và thư mục review

Chỉ cần chạy:

```bash
bash run_blind_quantization.sh
```

Mặc định chạy tuần tự hai tác vụ:

```bash
bash run_blind_quantization.sh --profile transfer --seeds 3407 --preservation-weights 0.5
bash run_blind_quantization.sh --watermark sleepermark
```

Mặc định là một seed 3407, preservation weight 0.5, thay cho ba mức
preservation cũ. [Protocol và lệnh chi tiết](survey/Scientific_Ablations_VI.md)
ghi rõ các nhánh, ngưỡng chất lượng, sweep và đánh giá trên key/checkpoint mới.
Các mục có ghi ngày cũ mô tả cấu hình lịch sử; bảng cấu hình đầu README được ưu tiên.
Mọi nhánh hiện dùng SSIM **trung bình ≥0.80** và các ngưỡng PSNR hiện có.
`science`/`transfer` không tự bật dual budget theo từng ảnh; nếu bật tường minh,
`budget_ssim` mặc định cũng là 0.80 với tối đa 10% ảnh vi phạm. Ngưỡng 0.80
làm kết quả dễ đạt quality gate hơn, không tự làm giảm TPR.

Mặc định research/science hiện là **2.000 update/nhánh**, gồm rounding, QAT
và fine-tune; tập natural TRAIN vẫn là **4.000 ảnh**, batch 4. Nhánh mới
`natural_teacher_rounding` dùng checkpoint FP32 đã chọn bằng SEARCH làm teacher,
rồi học rounding trên trọng số fingerprint gốc. Teacher được dùng chung với
đối chứng FP32, không train thêm một lần. Đây là nhánh tùy chọn trong `science`,
không có trong lệnh mặc định hoặc `transfer`.
Profile `pilot` vẫn có ngân sách nhỏ để kiểm tra nhanh.

Tất cả profile mặc định chỉ chạy **W4** cho các nhánh lượng tử hóa. W8 vẫn có
trong code dự phòng nhưng phải bật tường minh bằng `--bits 8` (suite: sau `--`).
Teacher và đối chứng FP32 chỉ chạy khi chọn profile/lệnh có chúng. Các script legacy cũng mặc
định W4; `WMQ_SEARCH_BITS` chỉ dùng để chủ động thay bitwidth trong legacy grid.

Để kiểm tra teacher → quantizer trước khi chạy bộ ablation lớn:

```bash
bash run_blind_quantization.sh --profile science
```

`science` có đối chứng FP32, fine-tune → RTN W4 và teacher → rounding W4;
vẫn 2.000 step và tự evaluate. Teacher mặc định được chọn trên SEARCH,
có thể đặt `--teacher-checkpoint-policy final` để dùng bước cuối định trước.
Profile này còn có random/DCT/contrastive ablation; không phải lệnh chạy mặc định.
`teacher_target_diagnostics.json` báo checkpoint và độ khác biệt teacher–model gốc
trên TRAIN/SEARCH. Đây là lựa chọn chỉ dựa trên ngân sách đã khai báo, không dùng
owner TEST để chọn teacher.
`natural_residual_qat_warm` khởi tạo mã W4 từ checkpoint `natural_residual` đã
chọn trên SEARCH, rồi học thêm code offset với cùng loss residual. Step 0 của
nhánh mới phải tái tạo chính xác mã W4 ban đầu; cả hai checkpoint được báo riêng,
không chọn bằng owner TEST. Đây là phép thử thêm, chưa được chứng minh tốt hơn.
`qk_rotation_ptq` được giữ trong profile `science` và lệnh tường minh, không còn
trong `transfer` vì run `152415` cho TPR/bit accuracy gần hệt fixed PTQ.
Nó là đối chứng model-only: quay cùng một cơ sở trực giao
trên Q/K của attention VAE (kể cả bias), kiểm tra đầu ra FP32 gần như không đổi,
rồi mới lượng tử hóa W4. Không dùng key, extractor hay ảnh natural để chọn phép
quay; kết quả của nó cần đo thực nghiệm, không giả định sẽ xóa watermark.

Toàn bộ source chuẩn nằm trong `src/`, gồm launcher, Python module, test, prompt và
requirements. Có thể chọn trực tiếp folder `src/` để đưa cho LLM khác review. Các
`run_blind_quantization.sh` ở root là wrapper ngắn chuyển nguyên tham số vào
`src/`; các launcher khác ở root có thể là symlink tương thích. Không có bản sao
logic chạy thí nghiệm.

Sau owner evaluation, mỗi run có thêm `result/`: các report JSON/CSV/JSONL ở cấp
run, toàn bộ `tradeoff/`, cùng selection/calibration gọn của từng branch. Checkpoint,
PNG và log update lớn vẫn ở vị trí chuẩn để tránh nhân đôi hàng chục GB. Không di
chuyển report gốc vì cơ chế freeze/integrity dùng đúng path và hash đó.

## Quality–evasion analysis và sweep (2026-09-23)

Owner evaluation tự tạo `tradeoff/points.csv`, `quality_thresholds.csv`, `pareto.csv`
và `quality_evasion.svg`. Tất cả checkpoint đã owner-evaluate đều được giữ, kể cả
quality failed; không tạo TPR giả cho các checkpoint chỉ có trong `search.json`.
Frontier tách bitwidth, comparison group và threat model; FP32 không được so như W4.
Trục evasion chính là tỷ lệ thoát detector **trong các ảnh reference đã được phát hiện**.
Bảng vẫn có evasion tuyệt đối và tỷ lệ đồng thời thoát detector + đạt ngưỡng từng ảnh.
Đây là các điểm đo và frontier mô tả, chưa phải đường cong liên tục đã được xác minh.

Với run đã có báo cáo, không cần chạy GPU lại:

```bash
python src/wmq_tradeoff.py --run output_attack/TEN_RUN
```

Nếu `tradeoff/` đã tồn tại, dùng `--output /path/to/new_analysis` để tránh ghi đè.
Các ngưỡng 0.80/0.82/0.85/0.86/0.90/0.95 trong bảng chỉ diễn giải lại cùng output.
`quality_policy=report` vốn không loại ứng viên quality failed. Nới ngưỡng riêng lẻ
không thay đổi model với policy này; SSIM 0.821 vẫn không đạt ngưỡng 0.85.

Để chạy một mức bảo toàn 0.5 với ngưỡng SSIM 0.85:

```bash
bash run_blind_quantization.sh --seeds 3407 --preservation-weights 0.5 --min-ssim 0.85
```

Suite khai báo trước một run. Khi truyền nhiều mức bằng `--preservation-weights`,
các run dùng cùng prompt/seed/calibration/bitwidth và thay đồng thời
`preserve_weight` và `qat_semantic_preserve_weight`, dùng low-pass preservation;
giữ report policy để lưu cả kết quả không đạt. FP32 control giữ cấu hình riêng,
các lần lặp control không phải bằng chứng độc lập. Có `suite_results.csv` và biểu đồ
`quality_evasion.svg` tổng hợp, tách theo seed. Thêm `--plan-only` để xem kế hoạch.
Không chọn winner bằng kết quả test rồi báo cùng tập đó là holdout chưa thấy.

`mechanism_analysis.json` được tổng hợp thành `tradeoff/layer_diagnostics.csv`.
Điểm rank dựa trên dấu/độ lớn tích vô hướng owner-error và quality-error, có floor
cho mẫu số; không dùng cosine nhỏ để suy ra tuyệt đối rằng layer không quan trọng.
Chỉ bốn ảnh, gradient tại endpoint và MSE quality không đủ chứng minh causal attribution
hoặc dự báo trực tiếp SSIM. Nếu có đủ layer với owner dot âm ổn định, báo cáo xuất
`owner_informed_steering_plan.json` cho thí nghiệm development riêng:

```bash
bash run_blind_quantization.sh --direct --bits 4 \
  --owner-informed-steering-plan output_attack/TEN_RUN/tradeoff/owner_informed_steering_plan.json
```

Các layer còn lại **vẫn được lượng tử hóa**, nhưng quantizer của chúng đóng băng ở
khởi tạo; chỉ layer trong plan được tối ưu ở các nhánh natural quantized. Benign PTQ
và FP32 control giữ vai trò đối chứng. Manifest/selection ghi rõ
`owner_informed_development_not_blind`: dữ liệu ranking đến từ victim test trước đó,
không phải surrogate độc lập. Không tự áp dụng plan vào run blind mặc định. Cần final
holdout mới để đánh giá; đây không phải implementation đầy đủ OS-MQ.

Table 1 của [Stable Signature is Unstable](https://arxiv.org/html/2405.07145v1)
báo SSIM 0.86 cho E-aware/E-agnostic. Đó là kết quả trong protocol của paper,
không phải chuẩn chất lượng phổ quát hay lý do đổi nhãn kết quả hiện tại thành thành công.

## Cấu hình mặc định sau run 20260922_023602

```bash
bash run_blind_quantization.sh
```

Lệnh này mặc định dùng profile `research`: W4, 4.000 ảnh natural train,
256 ảnh natural validation, 1.000 update/nhánh, batch train 4, ảnh natural 256px,
AdamW, warmup/cosine và LR fine-tune 5e-4. Các nhánh natural lượng tử hóa dùng
cùng low-pass preservation để so residual bật/tắt. Ngân sách lớn hơn pilot
32 ảnh/100 update; đây không phải bảo đảm attack thành công. Flag CLI ghi sau
vẫn override profile, ví dụ `--train-batch-size 1` khi thiếu VRAM.
`bash run_blind_quantization.sh --profile pilot` dùng cấu hình thử nhỏ.

Mỗi run gồm RTN, reconstruction chung/theo block, natural rounding có/không scale,
residual bật/tắt, QAT có/không scale, spectral reconstruction, GAN QAT và hai control
FP32. Các biến thể block/GAN/spectral là implementation thích nghi, **không phải
tái lập chính xác BRECQ, HiDDeN hoặc UnMarker**. VAE vẫn chạy FP32 với trọng số
low-bit dequantize; không dùng số liệu này để tuyên bố tốc độ kernel INT4.

Chọn checkpoint theo validation và đóng băng trước test. Nếu chọn checkpoint sớm,
profile research còn xuất/đánh giá checkpoint ở bước cuối với role `endpoint_control`;
không chọn lại theo TPR test. Các dòng `_final_test` dùng chung training với nhánh gốc,
không phải seed độc lập. `--no-evaluate-final` tắt đối chứng này.

Metric kém được ghi lại và tiếp tục. Nhánh runtime lỗi được ghi trong
`branch_failures.json` rồi tiếp tục nhánh khác; report có `status=branch_failures`,
không báo thành công giả. Lỗi setup/kiểm tra integrity vẫn có thể dừng run.
Owner evaluator tự chạy cuối cùng. Pool COCO mặc định gồm thêm 1.000 ảnh negative;
evaluator loại ảnh train/search theo hash và báo empirical FPR cùng Wilson CI.
Đây là FPR trên ảnh natural được khai báo không watermark, không thay cho mọi null
distribution. Đặt `WMQ_NEGATIVE_N=0` để bỏ hoặc `WMQ_NEGATIVE_IMAGES=/path` để dùng pool khác.

Để chạy một mức preservation weight 0.5 với seed 3407 và gom kết quả:

```bash
bash run_blind_quantization.sh
```

Suite mặc định dùng một seed 3407, preservation weight 0.5, ngưỡng SSIM 0.8,
W4 và profile `science`. Truyền nhiều giá trị `--preservation-weights` nếu chủ ý sweep. Xem
`output_attack/run_*/suite_results.csv`, `suite_status.json` và log từng seed.
Một run lỗi không ngăn các seed còn lại. `bash run_blind_quantization.sh --plan-only` tạo
folder `plan_*`, ghi `execution_mode=plan_only`, `suite_status=not_executed`
và không tải/chạy model. Folder này không chứa kết quả thực nghiệm. Các nhánh GAN có thêm discriminator compute, nên
cùng số update chưa có nghĩa cùng FLOPs; báo cả forward counts và thời gian.
Thay seed vẫn dùng cùng danh sách prompt: cần prompt/key holdout mới cho final paper.
Surrogate ownership transfer cần tài sản độc lập; code không thay nó bằng victim key.

## Blind: giữ model-only và thêm ảnh tự nhiên không ghép cặp

Launcher blind tự chuẩn bị dependencies trước khi kiểm tra GPU:

```bash
# auto: ưu tiên Conda wmq; không tìm thấy Conda thì dùng Python hiện tại
bash run_blind_quantization.sh

# Cài/chạy bằng Python hiện tại, kể cả khi máy có Conda
WMQ_ENV_MODE=current bash run_blind_quantization.sh

# Chọn chính xác interpreter hiện tại muốn dùng
WMQ_ENV_MODE=current WMQ_PYTHON=/path/to/python bash run_blind_quantization.sh
```

`WMQ_ENV_MODE=conda` yêu cầu Conda. Nhánh Conda luôn chọn `WMQ_CONDA_ENV`
(mặc định `wmq`), tạo Python 3.11 nếu môi trường chưa có, không lấy nhầm môi
trường `py`/`base` đang active. Nếu không có Conda, script không tải bộ cài Conda;
nó dùng `python`/`python3` trên PATH hoặc `WMQ_PYTHON` bạn chỉ định.

File mới **`prepare_blind_environment.py`** kiểm tra phiên bản, chạy
`<python> -m pip install --no-user -r requirements-wmq.txt` khi thiếu/sai phiên bản,
rồi kiểm tra import. Môi trường đã đủ thư viện sẽ không tải lại. Chế độ current
có thể thay đổi phiên bản package của môi trường hiện tại để khớp requirements.
Cần Python 3.10–3.12, mạng và quyền ghi môi trường; không dùng sudo hoặc tự vượt
cơ chế bảo vệ Python hệ thống. Driver NVIDIA vẫn phải được server cung cấp.
Log setup được lưu cùng log launcher trong `output_attack/`.

Chính sách tự cài này áp dụng cho **`run_blind_quantization.sh`**. Script fair
`run_malicious_quantization_watermarks.sh` vẫn dùng môi trường chuẩn bị sẵn như
hướng dẫn riêng bên dưới. Hãy copy cả repo lên server để có helper mới.

Luồng blind tự thử batch lớn (diffusion 8, VAE/metric 32), không hardcode tên GPU.
CUDA OOM ở inference/evaluation sẽ giảm một nửa batch và thử lại; seed từng ảnh
được tạo lại khi retry. Batch 1 vẫn OOM thì báo lỗi, không đổi dtype hoặc scope.
Latent/reference được cache trên GPU tối đa 16 GiB, chỉ khi sau khi cấp phát còn
ít nhất 50% tổng VRAM và tối thiểu 2 GiB trống; thiếu ngân sách thì giữ CPU.
Đây là ngân sách data cache, không bảo đảm mọi backward đều tránh OOM.

Training batch đặt bằng `--train-batch-size` (research: 4; pilot: 1).
Nhánh natural có thể thêm forward ảnh sinh cho preservation. Log giữ đủ từng update nhưng flush mỗi 50 bước,
cuối nhánh hoặc khi update lỗi. Nếu process bị kill đột ngột có thể mất phần log
chưa flush; dùng `--log-every 1` nếu cần ghi ngay. Runtime/cache/OOM backoff được
ghi trong report; batch và cấu hình được ghi manifest. Batching có thể tạo sai
số số học nhỏ so với chạy từng ảnh, không cam kết giống từng bit.

```bash
# Mặc định chạy bộ focused gồm 6 branch sau khi bỏ 3 cặp method/bit không hiệu quả
bash run_blind_quantization.sh

# Toàn bộ 24 branch dùng cho ablation đầy đủ
WMQ_METHOD_SET=full bash run_blind_quantization.sh --direct

# Chế độ đối chiếu từng ảnh, không cache GPU
WMQ_OWNER_BATCH_SIZE=1 bash run_blind_quantization.sh --direct \
  --gen-batch-size 1 --eval-batch-size 1 --data-cache cpu --log-every 1
```

Có thể đặt `--gen-batch-size`, `--eval-batch-size`, `--cache-max-gib` và
`--log-every` riêng. Batch owner evaluator đặt bằng `WMQ_OWNER_BATCH_SIZE`
(mặc định 0 = auto). Khi copy code lên server cần kèm **`wmq_runtime.py`**.
Launcher research dùng TF32 để tăng throughput trên GPU hỗ trợ và ghi cấu hình
vào manifest. Thêm `--cuda-math strict` nếu cần đối chiếu FP32 nghiêm ngặt.

Tối ưu cho GPU nhiều VRAM (bao gồm H200) áp dụng tự động với cùng lệnh chạy.
Không đổi batch training, số bước, loss hoặc dtype. Các scalar training được gom
vào một lần chuyển CPU; bỏ phép blur trung gian không dùng ở nhánh natural.
COCO tải đồng thời 8 ảnh, giữ nguyên danh sách và thứ tự theo seed. Mỗi ảnh đã
tải có receipt SHA256 để chạy lại tiếp tục phần còn thiếu, không tải lại ảnh đã
xác minh. Cache cũ chưa có receipt vẫn cần tải lại nếu chưa có provenance hoàn chỉnh.
Đổi số luồng tải bằng `WMQ_DOWNLOAD_WORKERS` (1–32, mặc định 8).
Mỗi request được thử tối đa 6 lần; các ảnh còn lỗi được thử lại trong tối đa 3
vòng mà không hủy những download khác đã thành công. Tiến độ được tính theo số
ảnh thực sự hoàn tất, không phụ thuộc thứ tự lấy mẫu.
Chưa benchmark toàn bộ suite trên H200 nên chưa có hệ số tăng tốc đo thực tế.

```bash
# Mặc định: cả hai hướng + tự tải checkpoint, ảnh COCO và tự đánh giá
bash run_blind_quantization.sh

# Dùng ảnh natural của bạn thay cho pool COCO tự tải
bash run_blind_quantization.sh --direct --natural-images /data/natural_images

# Chỉ chạy model-only nghiêm ngặt, không dùng ảnh/mạng perceptual bên ngoài để học
WMQ_MODEL_ONLY=1 bash run_blind_quantization.sh --direct
```

Thư mục natural chứa PNG/JPG/JPEG/WebP/BMP không watermark từ nguồn bạn chọn;
không cần ảnh sạch tương ứng với ảnh model sinh. Cần ít nhất `train-n + search-n`
file khác nội dung (mặc định 52). Runner chia tập và ghi hash vào manifest, chỉ
dùng encoder của model đã fingerprint; không dùng key/extractor để học quantizer.
Mặc định script tải metadata COCO và đúng số ảnh cần dùng từ bucket COCO công khai
(không tải toàn bộ archive ảnh), lưu pool theo count/seed, ghi URL/license/hash và
kiểm tra hash khi dùng lại. Không cần tài khoản dataset. COCO test2017 ở đây là
**dữ liệu calibration ngoài**; không phải tập test đánh giá attack.
Đây là quyền truy cập **model + dữ liệu natural + prior perceptual công khai**,
khác nhánh model-only. Mọi nhánh model-only hoàn tất trước khi nạp ảnh natural/LPIPS
để học; chúng không sử dụng dataset hoặc LPIPS trong loss/chọn checkpoint.
LPIPS owner evaluation vẫn được chạy sau freeze cho mọi nhánh.

Nhánh `natural_qat_purification` học tái tạo ảnh tự nhiên bằng hard-forward W4/STE. Mỗi
trọng số có một offset trong đơn vị mã lượng tử, mặc định giới hạn ở ±2 mã, nên có thể
đi xa hơn lựa chọn floor/ceil của `natural_rounding`. Loss bảo toàn chỉ so phần tần số
thấp của output sinh cùng latent; tránh dùng RGB MSE đầy đủ kéo decoder trở lại residual
đã fingerprint.
LPIPS AlexNet pretrained đóng băng; gradient truyền qua nó tới quantizer.
Chỉnh lambda bằng `--natural-perceptual-weight`; đặt 0 để ablation MSE-only.
Xem [hướng dẫn LPIPS chính thức](https://github.com/richzhang/PerceptualSimilarity)
và [COCO](https://cocodataset.org/#download). Các nhánh quantization chỉ học biến quantizer.
Đối chứng `natural_full_finetune` học toàn bộ decoder FP32, gồm bias/norm, và được ghi
riêng dưới role `finetune_control`; không tính là quantization attack.

**Lịch sử sau run 000857 (đã thay bằng profile ở đầu tài liệu):** từng mặc định 4 nhánh W4 `fixed_ptq`, `reconstruction`,
`natural_residual`, `natural_residual_qat` và 1 đối chứng `natural_full_finetune` FP32.
Nhánh residual dùng basis
patch từ natural TRAIN để tăng trọng số reconstruction theo hướng đã chọn, đồng thời
giảm phạt bảo toàn trong subspace đó. Basis đóng băng, không sử dụng detector/key.
Run 000857: residual BA 96,02%, TPR 98%, chỉ thêm 1/99 ảnh thoát detector.
QAT purification cũ quay về RTN. Các run tháng 9 mới hơn đã có kết quả trong thư mục báo cáo tương ứng.
Copy cả **`wmq_residual.py`** lên server. Lệnh vẫn là `bash run_blind_quantization.sh`.
Xem [phân tích kết quả, phương pháp và ablation](survey/Residual_Quantization_Revision_VI.md).
`--natural-methods natural_rounding natural_rounding_scale` khôi phục các nhánh
natural; `--methods` điều khiển riêng model-only. W8 chỉ chạy khi bật tường minh.
Model-only `rounding_scale` và sensitivity vẫn cần chọn riêng bằng `--methods`.

`natural_residual` có `--residual-weight` (1), `--residual-rank` (8),
`--residual-patch` (8), `--residual-patches-per-image` (256),
`--residual-preservation orthogonal|full` (orthogonal). Calibration JSON có captured
energy và split-half overlap; các chỉ số này không chứng nhận basis là watermark.
Đặt `--residual-weight 0 --residual-preservation full` để đối chiếu tương đương
natural_rounding. QAT purification có `--qat-max-code-shift` (2),
`--qat-trust-weight` (0.01) và `--qat-semantic-preserve-weight` (2). Không tăng số
step hay giảm bit mặc định cùng lúc với objective.

QAT dùng `--qat-lr 0.001`, độc lập `--lr 0.01` cho sigmoid rounding; FP32 pilot dùng
`--ft-lr 0.00001`, research dùng 0.0005. Warmup pilot là 10, research là 20 bước; cosine decay xuống 10%
learning rate đỉnh. `natural_residual_qat` dùng loss/basis/bảo toàn của residual,
thêm trust penalty code-offset; `--qat-trust-weight 0` là ablation cùng loss.
`natural_full_finetune` dùng natural MSE + LPIPS (`--ft-preserve-weight 0`);
`natural_gan_finetune` thêm discriminator. Cả hai chưa phải tái lập nguyên paper Duke.
Chi tiết lịch sử: [nghiên cứu và protocol](survey/Residual_QAT_Finetune_Revision_VI.md).

Lệnh mặc định là `bash run_blind_quantization.sh` với profile research ở đầu tài liệu.
Ví dụ chủ động chọn lại cấu hình nhỏ để đối chiếu run cũ:

```bash
WMQ_PROFILE=pilot bash run_blind_quantization.sh --direct --natural-train-n 256 --natural-search-n 64 \
  --steps 200 --qat-steps 1000 --ft-steps 1000 --eval-every 50
```

Ví dụ này cần 320 ảnh train/search và mặc định thêm 1.000 ảnh negative. Đây là run phát triển; prompt/test đã xem kết quả không trở
thành holdout mới chỉ vì đổi seed. Với paper, chốt cấu hình rồi dùng prompt/seed mới.

Mặc định `--quality-policy report`: chọn theo objective trên search; ngưỡng
PSNR/SSIM/TPR chỉ được ghi nhận, **không lọc candidate hay dừng vì không đạt ngưỡng**.
Mọi nhánh finite vẫn sinh test, được owner đánh giá và ghi CSV, kể cả khi cả baseline
và mọi candidate đều trượt. `--quality-policy constrained` khôi phục ưu tiên candidate
đạt quality gate nhưng vẫn không dừng khi không có candidate đạt.
Lỗi cấu hình, thiếu GPU/checkpoint, tải file hỏng hoặc đầu ra NaN không phải quality
failure hữu hạn; không được che giấu thành kết quả khoa học hợp lệ.

Trong `output_attack/<run>/`: `search.csv` lưu mọi candidate đã đánh giá;
`quality_summary.csv` lưu quality từng nhánh; `watermark_retention.csv` lưu owner
metrics và quality flags, kể cả `reference_valid=false`. `branches/*/updates.csv`
có từng thành phần loss. Toàn bộ dữ liệu nặng nằm trong một cây riêng như dưới đây.

Ảnh của run mới được lưu riêng, không nằm trong `output_attack`:

```text
output_attack/<run>/                    # CSV, JSON và log nhẹ để lấy kết quả
output_artifacts/models/marked_sd21/    # Pipeline Stable Signature công khai
output_artifacts/checkpoints/<run>/     # VAE checkpoint và quantizer từng nhánh
output_artifacts/images/<run>/          # PNG reference, pseudo-target và output
output_artifacts/datasets/              # Pool ảnh natural đã tải
output_artifacts/owner_assets/          # Extractor chỉ dùng sau freeze
output_artifacts/cache/                 # Hugging Face và Torch cache
```

Lệnh `bash run_blind_quantization.sh` vẫn tự evaluate. Evaluator đọc vị trí ảnh
từ manifest và kiểm tra hash như trước. Sau khi chạy xong, chỉ cần lấy CSV/JSON
để phân tích; không cần tải ảnh hoặc checkpoint về cùng báo cáo.
Đổi toàn bộ cây nặng bằng `WMQ_HEAVY_ROOT=/path/to/heavy`. Có thể override riêng
model/data/cache bằng `WMQ_MODEL_ROOT`, `WMQ_DATA_ROOT`, `WMQ_CACHE_ROOT`; đổi ảnh bằng
`WMQ_IMAGE_OUTPUT_ROOT=/path/to/images` hoặc
`--image-output /path/to/images/<run>` (thư mục run mới, không ghi đè).
Đổi nơi lưu checkpoint bằng `WMQ_CHECKPOINT_OUTPUT_ROOT=/path/to/checkpoints` hoặc
`--artifact-output /path/to/checkpoints/<run>`. Manifest lưu cả hai vị trí; nếu di chuyển
checkpoint trước khi chạy evaluator thủ công, truyền `--artifact-root /new/path/<run>`.
Nếu chuyển ảnh sang vị trí khác trước khi evaluate, truyền
`--image-root /new/path/<run>` cho `evaluate_blind_watermark.py`.
Run cũ vẫn được đọc theo cấu trúc cũ; thay đổi này không di chuyển ảnh của run đã có.

Mọi run mới xuất ba loại ảnh: `marked_reference_test`, `pseudo_target_test` và
các output quantizer `*_test`. Owner đánh giá cả ba sau khi freeze mọi lựa chọn.
`watermark_retention.csv` có double-tail TPR/evasion, quality và chỉ báo texture;
`report.json` có metric chi tiết từng ảnh. `reconstruction` là đối chứng mặc định
không smoothing. Sensitivity chỉ đo proxy thị giác, không được gọi là độ nhạy ownership.

Detector mặc định **double-tail**, hiệu chỉnh ngưỡng nguyên theo FPR **tổng**:
48 bit/FPR 0.001 phát hiện khi khớp >=36 hoặc <=12; FPR 0.0001 dùng >=38 hoặc <=10.
Dùng `FPR=0.0001 bash run_blind_quantization.sh ...` để đổi FPR trước chạy.
`DETECTOR=single` chỉ dành cho đối chiếu cũ. Baseline không đạt vẫn có báo cáo;
FPR bất khả thi với độ dài key là cấu hình sai và bị từ chối.

Blur có thể giữ watermark hoặc làm mất texture. Metric trực tiếp và đối chứng
giúp kiểm tra giả thuyết, **không bảo đảm attack thành công**. Hướng natural cũng
cần kiểm chứng. Chi tiết protocol tại
[survey/Blind_Quantization_Stable_Signature.md](survey/Blind_Quantization_Stable_Signature.md).

Script `run_malicious_quantization_watermarks.sh` đánh giá khả năng giữ watermark của **Stable Signature** và **AquaLoRA** sau lượng tử hóa. Mặc định dùng `RUN_PROTOCOL=fair_w4a16` để so sánh cùng bitwidth/phạm vi; quy trình grid rộng trước đây còn ở `RUN_PROTOCOL=legacy_grid`:

1. Grid search các recipe PTQ theo bit-width, clipping và nhóm layer.
2. Tối ưu tham số quantizer bằng autograd và straight-through estimator (STE), hoặc dùng zeroth-order search làm ablation.
3. Đánh giá recipe đã chọn trên tập test tách biệt bằng bit accuracy, TPR, PSNR, LPIPS và prediction NMSE.

Đây là simulated weight-only PTQ: trọng số được đưa lên lưới số nguyên low-bit rồi dequantize để chạy bằng CUDA FP16. Kết quả phản ánh sai số lượng tử hóa, không phải tốc độ của INT4/INT8 kernel. Copy cả `wmq_baselines.py`, `wmq_diagnostics.py` cùng script và requirements sang server.

## Tiếp tục chạy khi metric không đạt

Mặc định `CLEAN_POLICY=report`: baseline có bit accuracy/TPR thấp vẫn chạy hết
quantization và test. PSNR/LPIPS/NMSE không đạt được ghi nhận bằng `feasible=false`,
không làm dừng experiment. Khi cả grid legacy đều trượt, script chọn candidate có
NMSE thấp nhất trong các candidate có metric finite, rồi tiếp tục refinement/test;
report ghi `grid_diagnostic_fallback`. Ngưỡng vẫn dùng để ưu tiên candidate hợp lệ,
không bị hạ xuống để biến kết quả thất bại thành thành công.

- Log toàn bộ stdout/stderr: `wmq_runs/logs/wmq_<thời gian>_<mã>.log`; đường dẫn
  được in khi bắt đầu. Đổi thư mục bằng `WMQ_LOG_DIR`.
- Cảnh báo tập trung: `$WMQ_OUTPUT/quality_warnings.jsonl`, mặc định
  `wmq_runs/output/run_<thời gian>_<mã>/quality_warnings.jsonl`. Mỗi dòng có stage, metric, ngưỡng và action.
- Kết quả chi tiết vẫn ở `report.json`/`comparison.csv`. `status=complete` nghĩa là
  chạy xong; kiểm tra thêm `baseline_valid`, `feasible`, `quality_valid` để biết chất lượng.
- Với `run_blind_quantization.sh`, log console vẫn ở `output_attack/*.log` và cảnh
  báo ở `<thư mục run>/quality_warnings.jsonl`; tất cả nhánh trượt vẫn có ảnh/report.

Chỉ đặt `CLEAN_POLICY=strict` khi chủ động muốn dừng nhánh có baseline không đạt.
`BASELINE_CHECK_ONLY=1` vẫn chỉ kiểm tra baseline như tên gọi. Lỗi thiếu thư viện,
checkpoint hỏng, cấu hình sai hoặc không còn candidate có metric hữu hạn vẫn báo
lỗi; đó không phải trường hợp metric hợp lệ nhưng thấp hơn ngưỡng chất lượng.

## So sánh W4A16 mới

Trong job đã activate Conda, chạy:

```bash
bash run_malicious_quantization_watermarks.sh
```

Script tự tạo thư mục mới `wmq_runs/output/run_<thời gian>_<mã>` cho mỗi lần chạy và in đường dẫn ra terminal; không cần đặt `CLEAN_POLICY` hoặc `WMQ_OUTPUT`. Nếu muốn tự chọn đường dẫn, vẫn có thể đặt `WMQ_OUTPUT`; script không ghi đè kết quả fair đã tồn tại. Môi trường Conda cần được activate sẵn như hướng dẫn bên dưới. Không cần thêm dependency ngoài `requirements-wmq.txt`.

| Phương pháp | VAE Stable Signature | UNet AquaLoRA |
|---|---|---|
| `rtn_w4a16` | Làm tròn thông thường | Làm tròn thông thường |
| `grid_w4a16` | Chọn clipping 1,0/0,75 trên search | Tương tự |
| `gradient_w4a16` | Tối ưu theo watermark; báo rõ nếu fallback | Tương tự |
| `adaround_vae` | Học làm tròn từng weight theo đầu ra layer | — |
| `brecq_vae_adapted` | Hiệu chỉnh theo block | — |
| `qdiff_unet_adapted` | — | Hiệu chỉnh theo block, nhiều timestep, tách grid cho shortcut concat |

Mọi phương pháp trong cùng watermark đều lượng tử hóa **cùng toàn bộ target**: trọng số Conv/Linear của VAE decoder + post-quant conv, hoặc của UNet. Encoder VAE, bias, normalization và watermark extractor không được lượng tử hóa. Trọng số dùng 4 bit (grid đối xứng -7..7), activation giữ FP16; scale lưu FP32. Q-Diffusion adaptation có thêm scale riêng cho hai phần input concat, được ghi trong `shortcut_splits`; cùng bitwidth không có nghĩa overhead scale giống nhau. Tái tạo block chạy FP32 để ổn định gradient, sau đó weight dequantized được đưa về dtype inference.

Các baseline reconstruction là **bản thích nghi cục bộ cho Diffusers**, không phải chạy nguyên repo hoặc tái lập nguyên số liệu paper. `AdaRound` dùng stretched-sigmoid rounding và regularization; BRECQ adaptation dùng MSE đầu ra block, không dùng Fisher weighting; Q-Diffusion adaptation kết hợp block reconstruction với replay nhiều timestep và split shortcut trên Conv1/Conv-shortcut của up-block. Input calibration của từng block được thu tuần tự khi các block trước đã được lượng tử hóa. Không baseline reconstruction nào dùng key/loss watermark. Chi tiết nằm trong `wmq_baselines.py` và từng report. Nguồn: [AdaRound](https://arxiv.org/abs/2004.10568), [BRECQ](https://arxiv.org/abs/2102.05426), [Q-Diffusion](https://arxiv.org/abs/2302.04304).

Ba phần dữ liệu tách biệt theo chỉ số và seed: `CALIB_N` prompt calibration, `SEARCH_N` prompt chọn cấu hình, `TEST_N` prompt test. Mọi phương pháp tái sử dụng đúng các phần này. `PROMPT_FILE` cần ít nhất `CALIB_N + SEARCH_N + TEST_N` dòng. Manifest lưu prompt/seed, scheduler, target names, số tham số và hash checkpoint. Cấu hình cuối được chọn bằng search/calibration; test chỉ đánh giá. Các phương pháp có ngân sách tối ưu khác nhau, được báo cáo qua số step/update; đây không phải so sánh cùng chi phí tính toán.

Grid, refinement và fallback đều được kiểm tra **PSNR ≥25, LPIPS ≤0,15 và calibration NMSE ≤0,02**. Nếu không có candidate đạt, report vẫn lưu kết quả với `feasible=false`; không gọi đó là cấu hình thành công. Bản gradient có `selected_source=grid_fallback` khi không chọn được refinement hợp lệ. File `gradient_updates.json` được ghi mỗi bước, gồm số update hợp lệ, bị bỏ qua và lý do; không tính update NaN đã rollback là thành công.

Kiểm tra chất lượng watermark trước lượng tử hóa trên GPU:

```bash
BASELINE_CHECK_ONLY=1 WMQ_OUTPUT="$PWD/results/clean_check" \
  bash run_malicious_quantization_watermarks.sh
```

Khác với `WMQ_CHECK_ONLY=imports`, lệnh này **tải model và sinh ảnh**. Ngưỡng clean search mặc định là bit accuracy ≥0,80 và TPR ≥0,90. Đây là quality gate được đặt trước thí nghiệm, không phải bằng chứng checkpoint sai nếu không đạt. `clean_validation.json` luôn được ghi. Trong run thông thường, `CLEAN_POLICY=report` (mặc định) tiếp tục khi baseline không đạt và giữ `baseline_valid=false` trong kết quả. Chỉ khi chủ động đặt `CLEAN_POLICY=strict` thì nhánh fair không đạt mới dừng riêng với `invalid_clean_baseline`. Có thể đặt `CLEAN_MIN_BITACC`/`CLEAN_MIN_TPR` trước run theo protocol nghiên cứu.

| Biến mới | Mặc định | Ý nghĩa |
|---|---:|---|
| `RUN_PROTOCOL` | `fair_w4a16` | Hoặc `legacy_grid` cho tìm kiếm rộng cũ |
| `RECON_STEPS` | 200 | Số update mỗi layer/block |
| `RECON_LR` | 0,001 | Learning rate cho reconstruction |
| `RECON_CACHE_MB` | 512 | Giới hạn cache CPU cho một block |
| `CLEAN_POLICY` | `report` | Metric thấp vẫn chạy; `strict` để chủ động chặn baseline không đạt |
| `BASELINE_CHECK_ONLY` | 0 | Đặt 1 để chỉ kiểm tra clean trên GPU |

Trong fair protocol, `REFINE_N` và `MAX_CANDIDATES` không dùng: gradient dùng toàn bộ search split, grid luôn là W4 trên toàn target với hai clipping. Zeroth-order vẫn có ở `RUN_PROTOCOL=legacy_grid`. `REFINE_MODE=none` bỏ dòng gradient nhưng vẫn chạy các baseline reconstruction.

Kết quả mới:

```text
results/w4a16_seed3407/
├── comparison.csv
├── comparison_summary.json
└── <watermark>/fair_w4a16/
    ├── manifest.json
    ├── clean_validation.json
    ├── clean_test.json
    ├── grid_search.json
    ├── report.json
    ├── clean_search/
    ├── clean_test/
    └── <method>/
        ├── report.json
        ├── attacked_test/
        ├── reconstruction.json       # baseline reconstruction
        ├── gradient_updates.json     # gradient, ghi mỗi bước
        └── gradient_optimization.csv # gradient, ghi cuối loop
```

Nếu clean gate không đạt, chưa có các file search/test/method tương ứng; summary ghi `incomplete`. Các report method đã hoàn thành được lưu ngay, không phải chờ cả AquaLoRA kết thúc. Reconstruction replay model một lần cho mỗi unit/tập calibration, nên chi phí có thể lớn. Bắt đầu bằng run nhỏ trước khi chạy dataset đầy đủ:

```bash
SEARCH_N=2 TEST_N=2 CALIB_N=1 CALIB_TIMESTEPS=2 STEPS=4 \
RECON_STEPS=2 GRAD_STEPS=2 GEN_BATCH_SIZE=1 CALIB_BATCH_SIZE=1 \
WMQ_OUTPUT="$PWD/results/w4a16_smoke" bash run_malicious_quantization_watermarks.sh
```

Kiểm thử code không cần GPU/checkpoint:

```bash
python -m unittest discover -s tests -v
```

Các test chạy Conv/Linear, UNet và VAE Diffusers nhỏ với trọng số ngẫu nhiên; kiểm tra grid, backward, coverage, input concat, rollback NaN/Inf, fallback NMSE và báo cáo/split. Chúng không chứng minh chất lượng watermark hay mức bộ nhớ trên checkpoint thật; cần smoke test trên GPU server.

## 1. Cài Conda environment trên login node (một lần)

Script chạy chỉ sử dụng Conda environment đã được kích hoạt; không tạo môi trường hoặc cài package trong job. Cần copy cả repo (bao gồm `requirements-wmq.txt`) vào filesystem mà login node và compute node cùng truy cập được. Conda environment cũng phải nằm trên filesystem dùng chung của cụm.

Trên login node, vào thư mục repo rồi chạy:

```bash
conda create -n wmq python=3.11 pip -y
conda activate wmq
unset PYTHONPATH PYTHONHOME PYTHONUSERBASE PIP_TARGET PIP_PREFIX PIP_USER
export PYTHONNOUSERSITE=1
export PIP_CONFIG_FILE=/dev/null PIP_REQUIRE_VIRTUALENV=false
python -m pip install -r requirements-wmq.txt
WMQ_CHECK_ONLY=imports bash run_malicious_quantization_watermarks.sh
```

Nếu đã có môi trường `wmq` dành riêng cho dự án, bỏ lệnh `conda create` và activate nó trước khi cài. Không cài vào `base` hoặc môi trường dùng chung với dự án khác. Conda quản lý Python/môi trường; pip cài các phiên bản thư viện vào chính môi trường đó theo [hướng dẫn Conda về dùng pip trong environment](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html#using-pip-in-an-environment).

Login node không cần GPU để cài wheel CUDA và kiểm tra import. Giữ wheel CUDA cho compute node, không chuyển sang wheel CPU chỉ vì login node không có GPU. Cặp Torch/TorchVision lấy từ [hướng dẫn PyTorch](https://pytorch.org/get-started/previous-versions/#v271). File `requirements-wmq.txt` đã chứa nguồn tải và phiên bản CUDA, nên chỉ cần một lệnh pip. Với GPU/driver cũ, đổi đồng thời `cu128` trong URL và hai phiên bản Torch/TorchVision của file này thành `cu118` hoặc `cu126`; Blackwell cần `cu128`. Máy chạy cần Linux x86_64, driver NVIDIA tương thích và dung lượng cho thư viện, model, kết quả.

## 2. Dùng environment đã cài trong job

Trong file submit job đang dùng của server, giữ các dòng khai báo scheduler/tài nguyên và thêm phần sau vào thân job trước lệnh chạy:

```bash
source /duong/dan/miniconda3/etc/profile.d/conda.sh
conda activate wmq
cd /duong/dan/Malicious_quantization
bash run_malicious_quantization_watermarks.sh
```

Thay hai đường dẫn trên bằng đường dẫn thực tế trên server. Trên login node, `conda info --base` cho biết thư mục cài Conda. Nếu cụm yêu cầu `module load` để cung cấp Conda, dùng cơ chế đó theo hướng dẫn của cụm. Kích hoạt tường minh trong job giúp chọn đúng môi trường, kể cả khi scheduler không truyền toàn bộ môi trường của login shell. Không chạy lệnh cài package trong job và không sửa môi trường khi job khác đang dùng nó.

Chưa biết server dùng Slurm, PBS hay scheduler khác nên ví dụ trên chỉ là phần thân job; dùng cấu hình GPU/partition/queue của server. Giữ `CUDA_VISIBLE_DEVICES` do scheduler cấp.

## 3. Kiểm tra trước khi chạy thí nghiệm

Trên login node (không yêu cầu GPU):

```bash
conda activate wmq
WMQ_CHECK_ONLY=imports bash run_malicious_quantization_watermarks.sh
```

Trong job đã được cấp GPU:

```bash
WMQ_CHECK_ONLY=1 bash run_malicious_quantization_watermarks.sh
```

Script chạy `pip check`, kiểm tra phiên bản, toàn bộ import và TorchVision NMS; chế độ `1` còn thử forward/backward CUDA FP16 với convolution và attention. Hai chế độ này không tải model hoặc chạy thí nghiệm. Phiên bản thực tế được lưu trong `environment.freeze.txt` tại thư mục kết quả được in khi bắt đầu.

Bộ thư viện đã được kiểm tra trước đó trên Python 3.12/Linux CPU: import, NMS, AquaLoRA decoder/Mapper, LPIPS với backbone ngẫu nhiên và UNet nhỏ forward/backward đều qua. Máy phát triển hiện không có Conda/GPU, nên chưa xác minh cài đặt Conda hoặc thí nghiệm GPU đầy đủ trên server.

Chạy thí nghiệm vẫn cần truy cập các model/checkpoint. Nếu compute node không có Internet, phải chuẩn bị cache/model trước trên filesystem dùng chung; kiểm tra import không tải sẵn các tài nguyên này. Nếu model yêu cầu xác thực, đặt `HF_TOKEN` theo quyền truy cập tài khoản.

Script tự chọn profile `large-memory` khi GPU nhìn thấy có ít nhất 80 GiB; GPU nhỏ hơn dùng cấu hình tiết kiệm bộ nhớ.

Model mặc định:

- Stable Signature: `stabilityai/stable-diffusion-2-1-base`, với fallback `sd2-community/stable-diffusion-2-1-base`.
- AquaLoRA: `stable-diffusion-v1-5/stable-diffusion-v1-5`.
- AquaLoRA assets: `georgefen/AquaLoRA-Models/ppft_trained`.

## 4. Chạy cấu hình mặc định

```bash
conda activate wmq
bash run_malicious_quantization_watermarks.sh
```

Trên RTX PRO 6000 Blackwell 96 GB, log đầu chương trình phải hiển thị gần như sau:

```text
large_memory=True, gen_batch=16, metric_batch=32, calib_batch=2,
pristine_on_gpu=True, joint_group_grad=True, attention_slicing=False
```

Không cần đặt thủ công các biến này. Nếu server đang dùng MIG và process chỉ nhìn thấy một partition dưới 80 GiB, profile sẽ tự chuyển sang chế độ tiết kiệm bộ nhớ dựa trên dung lượng CUDA thực tế mà process nhìn thấy.

Mặc định script sử dụng:

| Biến | Giá trị | Ý nghĩa |
|---|---:|---|
| `SEARCH_N` | 20 | Số ảnh dùng để chọn recipe PTQ |
| `TEST_N` | 100 | Số ảnh test độc lập |
| `STEPS` | 25 | Số bước DDIM |
| `GUIDANCE` | 7.0 | Classifier-free guidance scale |
| `MIN_PSNR` | 25.0 | Ngưỡng PSNR tối thiểu |
| `MAX_LPIPS` | 0.15 | Ngưỡng LPIPS tối đa |
| `MAX_PRED_NMSE` | 0.02 | Ngưỡng prediction/decode NMSE |
| `REFINE_OPTIMIZER` | `gradient` | Autograd + STE |
| `REFINE_MODE` | `full` | Tối ưu scale, zero-point và rounding |
| `GRAD_STEPS` | 40 | Số gradient update |
| `GRAD_EVAL_EVERY` | 4 | Khoảng cách giữa các hard evaluation |
| `REFINE_N` | 8 | Số ảnh feedback trong refinement |
| `CALIB_N` | 2 | Số trajectory/latent calibration |
| `CALIB_TIMESTEPS` | 5 | Số timestep lấy trên mỗi trajectory |
| `GEN_BATCH_SIZE` | 16 trên GPU ≥80 GiB | Batch sinh ảnh |
| `METRIC_BATCH_SIZE` | 32 trên GPU ≥80 GiB | Batch LPIPS/PSNR |
| `CALIB_BATCH_SIZE` | 2 trên GPU ≥80 GiB | Batch trajectory calibration |
| `KEEP_PRISTINE_ON_GPU` | 1 trên GPU ≥80 GiB | Tránh copy state qua PCIe mỗi candidate |
| `JOINT_GROUP_GRAD` | 1 trên GPU ≥80 GiB | Cập nhật đồng thời mọi semantic group |
| `USE_ATTENTION_SLICING` | 0 trên GPU ≥80 GiB | Không chia attention thành lát nhỏ |

Ở `RUN_PROTOCOL=legacy_grid`, grid search mặc định thử 10 recipe cho mỗi watermark:

- Bit-width: 4 bit; các bitwidth khác chỉ chạy nếu đặt `WMQ_SEARCH_BITS` tường minh.
- Clipping: 1.0 và 0.75.
- Phạm vi: toàn carrier hoặc từng nhóm layer.

Vì mỗi recipe phải sinh ảnh, cấu hình mặc định có thể chạy lâu.

## 5. Chạy cấu hình dùng cho paper

Nên dùng prompt cố định, lưu seed/config và chạy nhiều seed độc lập:

```bash
CUDA_VISIBLE_DEVICES=0 \
WMQ_ROOT="$PWD/runs/seed3407" \
PROMPT_FILE="$PWD/prompts/coco_prompts.txt" \
BASE_SEED=3407 \
SEARCH_N=20 \
TEST_N=100 \
STEPS=25 \
MAX_CANDIDATES=0 \
REFINE_OPTIMIZER=gradient \
REFINE_MODE=full \
GRAD_STEPS=80 \
GRAD_EVAL_EVERY=4 \
GRAD_LR=0.03 \
REFINE_N=20 \
CALIB_N=4 \
CALIB_TIMESTEPS=8 \
MIN_PSNR=25.0 \
MAX_LPIPS=0.15 \
MAX_PRED_NMSE=0.02 \
bash run_malicious_quantization_watermarks.sh
```

`PROMPT_FILE` phải chứa ít nhất `CALIB_N + SEARCH_N + TEST_N` prompt cho fair protocol (`SEARCH_N + TEST_N` cho legacy). File text dùng một prompt trên mỗi dòng. JSON có thể là:

```json
[
  "a red fox in a snowy forest",
  "a lighthouse above a stormy sea"
]
```

hoặc:

```json
{
  "prompts": [
    "a red fox in a snowy forest",
    "a lighthouse above a stormy sea"
  ]
}
```

Nếu không đặt `PROMPT_FILE`, script tự tạo tập prompt xác định để chạy thử nghiệm sơ bộ.

## 6. Ablation

### Grid PTQ, không refinement

```bash
REFINE_MODE=none \
bash run_malicious_quantization_watermarks.sh
```

### Chỉ tối ưu scale

```bash
REFINE_OPTIMIZER=gradient \
REFINE_MODE=scale \
bash run_malicious_quantization_watermarks.sh
```

### Tối ưu scale và zero-point

```bash
REFINE_OPTIMIZER=gradient \
REFINE_MODE=scale_zero \
bash run_malicious_quantization_watermarks.sh
```

### Tối ưu đầy đủ bằng gradient

```bash
REFINE_OPTIMIZER=gradient \
REFINE_MODE=full \
bash run_malicious_quantization_watermarks.sh
```

### Zeroth-order baseline

```bash
REFINE_OPTIMIZER=zeroth \
RUN_PROTOCOL=legacy_grid \
REFINE_MODE=full \
REFINE_ITERS=100 \
bash run_malicious_quantization_watermarks.sh
```

Mỗi ablation nên dùng một `WMQ_ROOT` riêng để tránh ghi đè output:

```bash
WMQ_ROOT="$PWD/runs/grid" REFINE_MODE=none bash run_malicious_quantization_watermarks.sh
WMQ_ROOT="$PWD/runs/scale" REFINE_MODE=scale bash run_malicious_quantization_watermarks.sh
WMQ_ROOT="$PWD/runs/scale_zero" REFINE_MODE=scale_zero bash run_malicious_quantization_watermarks.sh
WMQ_ROOT="$PWD/runs/full" REFINE_MODE=full bash run_malicious_quantization_watermarks.sh
WMQ_ROOT="$PWD/runs/zeroth" REFINE_OPTIMIZER=zeroth REFINE_MODE=full bash run_malicious_quantization_watermarks.sh
```

## 7. Các biến cấu hình

### Đường dẫn và môi trường

| Biến | Mặc định | Mô tả |
|---|---|---|
| `WMQ_ROOT` | Thư mục script + `/wmq_runs` | Thư mục gốc của một lần chạy |
| `WMQ_OUTPUT` | Tự tạo `$WMQ_ROOT/output/run_<thời gian>_<mã>` | Thư mục kết quả riêng mỗi lần chạy |
| `WMQ_CACHE` | `$WMQ_ROOT/hf_cache` | Hugging Face cache |
| `CONDA_PREFIX` | Do `conda activate` đặt | Environment đã cài trên login node |
| `WMQ_CHECK_ONLY` | `0` | `1`: import + CUDA; `imports`: chỉ import |
| `TORCH_HOME` | `$WMQ_ROOT/torch_cache` | Cache trọng số TorchVision/LPIPS |
| `CUDA_VISIBLE_DEVICES` | không đặt | Chọn GPU sẽ chạy |

Dependency trực tiếp được ghim trong `requirements-wmq.txt`. Script chỉ kiểm tra môi trường Conda đã activate và chạy; không tự cài đặt. Dependency gián tiếp thực tế được ghi trong `environment.freeze.txt`.

### Model và key

| Biến | Mô tả |
|---|---|
| `SS_MODEL` | Stable Signature backbone chính |
| `SS_MODEL_FALLBACK` | Backbone fallback |
| `AQUA_MODEL` | SD1.5 backbone cho AquaLoRA |
| `SS_KEY` | Stable Signature key 48 bit |
| `AQUA_KEY` | AquaLoRA message 48 bit |
| `AQUA_FOLDER` | `ppft_trained` hoặc checkpoint folder khác |

`SS_KEY` và `AQUA_KEY` phải là chuỗi nhị phân đúng 48 ký tự.

### Gradient optimizer

| Biến | Mặc định | Mô tả |
|---|---:|---|
| `GRAD_STEPS` | 40 | Tổng số Adam update |
| `GRAD_LR` | 0.03 | Learning rate của quantizer controls |
| `GRAD_EVAL_EVERY` | 4 | Hard evaluation sau mỗi số bước này |
| `ROUND_TEMPERATURE` | 0.10 | Nhiệt độ sigmoid surrogate cho STE rounding |
| `GRAD_PRED_WEIGHT` | 1.0 | Trọng số prediction/decode NMSE trong gradient loss |
| `GRAD_IMAGE_WEIGHT` | 0.10 | Trọng số image MSE trong gradient loss |
| `TPR_LOSS_WEIGHT` | 0.25 | Trọng số TPR trong hard objective |
| `PRED_LOSS_WEIGHT` | 0.10 | Trọng số prediction NMSE trong hard objective |
| `REFINE_SEED` | 2026 | Seed của zeroth-order optimizer |

Ở RTX PRO 6000 Blackwell 96 GB, mặc định mỗi bước mở graph joint cho toàn bộ semantic group. Trên GPU dưới 80 GiB, script chuyển sang coordinate-gradient: mỗi bước mở graph cho một group, còn các group khác giữ hard-quantized để giảm VRAM.

## 8. Giới hạn và xấp xỉ của gradient refinement

### Coordinate-gradient theo group và timestep

Trong portable profile, mỗi gradient step chỉ mở graph cho:

- một semantic group của carrier;
- một calibration state tại một timestep;
- các group còn lại ở trạng thái hard-quantized hiện tại.

Khi recipe ban đầu chọn toàn carrier, portable profile tối ưu bốn group theo round-robin. Với `GRAD_STEPS=40`, mỗi group nhận khoảng 10 update. Large-memory profile cập nhật cả bốn group ở mỗi bước, nên mỗi group nhận đủ 40 update. Nếu recipe ban đầu chỉ chọn một group thì hai profile tương đương nhau.

Large-memory profile là joint-gradient theo group nhưng mỗi update vẫn dùng một timestep-batch. Portable profile là coordinate-gradient theo cả group và timestep. Khi viết paper phải ghi profile đã dùng và báo cáo ablation theo `GRAD_STEPS`, `CALIB_TIMESTEPS` và `JOINT_GROUP_GRAD`.

### Single-step predicted-x0 proxy

Đối với AquaLoRA, script không backpropagate qua toàn bộ DDIM trajectory. Tại mỗi calibration state, nó:

1. Chạy UNet đã quantize tại `(x_t, t, c)`.
2. Tạo guided noise prediction.
3. Ước lượng `x_0` trực tiếp từ `x_t` và noise prediction.
4. Decode `x_0` bằng VAE cố định.
5. Backpropagate watermark loss từ AquaLoRA decoder.

Thuật ngữ phù hợp là **single-step predicted-x0 differentiable proxy** hoặc **single-timestep reconstruction proxy**. Không gọi đây là full-trajectory differentiable optimization. Cũng không nên gọi là “linearized proxy” trừ khi phương pháp được bổ sung phép khai triển Taylor hoặc Jacobian rõ ràng.

Đối với Stable Signature, carrier nằm trong VAE decoder nên gradient đi trực tiếp qua VAE decoder và Stable Signature extractor trên latent cuối; nhánh này không cần xấp xỉ noise trajectory.

Sau mỗi `GRAD_EVAL_EVERY` bước, quantizer được materialize thành hard low-bit weights và đánh giá bằng quá trình sinh ảnh DDIM hoàn chỉnh. Checkpoint cuối chỉ được chọn từ các hard evaluation thỏa PSNR, LPIPS và prediction/decode NMSE. Bước này kiểm tra proxy bằng hành vi end-to-end nhưng không biến quá trình huấn luyện thành full-trajectory backpropagation.

## 9. Output

Kết quả nằm trong `wmq_runs/output/run_<thời gian>_<mã>` hoặc `WMQ_OUTPUT` nếu tự đặt. Cây dưới đây là **legacy_grid**; output fair mặc định được mô tả ở đầu tài liệu:

```text
output/
├── stable_signature/
│   ├── clean_test/
│   ├── attacked_test/
│   ├── search.csv
│   ├── gradient_optimization.csv
│   ├── quantizer_optimization.csv       # chỉ có với zeroth-order
│   └── report.json
├── aqualora/
│   ├── clean_test/
│   ├── attacked_test/
│   ├── search.csv
│   ├── gradient_optimization.csv
│   ├── quantizer_optimization.csv       # chỉ có với zeroth-order
│   └── report.json
├── summary.json
└── watermark_retention.csv
```

Các file quan trọng:

- `search.csv`: toàn bộ 50 recipe PTQ và kết quả trên search split.
- `gradient_optimization.csv`: proxy loss, gradient norm và hard evaluation theo step.
- `report.json`: recipe được chọn và kết quả test của từng watermark.
- `summary.json`: cấu hình và kết quả đầy đủ của cả hai watermark.
- `watermark_retention.csv`: bảng gọn để plot hoặc nhập vào pandas/R/Excel.

Không dùng các chỉ số trên search/refinement split làm kết quả cuối. Báo cáo paper phải lấy `attacked`, `paired_quality` và retention từ test split trong `report.json` hoặc `watermark_retention.csv`.

## 10. Checklist trước khi lấy kết quả paper

- [ ] Smoke test chạy qua cả Stable Signature và AquaLoRA.
- [ ] Clean bit accuracy của cả hai watermark lớn hơn hoặc bằng 0.80.
- [ ] Dùng prompt file cố định có ít nhất `SEARCH_N + TEST_N` mẫu.
- [ ] Search, refinement và test dùng các seed tách biệt theo script.
- [ ] `MAX_CANDIDATES=0` để không cắt recipe.
- [ ] Chạy đủ các ablation `none`, `scale`, `scale_zero`, `full`, `zeroth`.
- [ ] Chạy nhiều `BASE_SEED` và lưu mỗi run trong `WMQ_ROOT` riêng.
- [ ] Báo cáo bit accuracy, TPR, PSNR, LPIPS và prediction NMSE.
- [ ] Kiểm tra `quantizer_grad_norm` khác 0 trong `gradient_optimization.csv`.
- [ ] Ghi rõ dùng joint-group hay coordinate-group gradient và single-step predicted-x0 proxy.
- [ ] Không tuyên bố đã backpropagate qua toàn bộ diffusion trajectory.
- [ ] Chỉ kết luận từ test split, không chọn kết quả theo test split.

## Stable Signature: pilot blind với hai mức quyền truy cập

Nhánh `run_blind_quantization.sh` dùng `wmq_blind.py` để kiểm tra model-only và
model + natural. Các nhánh model-only không dùng key, extractor, detector feedback,
model sạch, ảnh sạch bên ngoài hay pretrained perceptual prior trong optimization.
Các nhánh natural dùng thêm ảnh công khai và LPIPS như mô tả đầu tài liệu. Đây là **Pilot 0 mở rộng**;
chưa phải OS-MQ hoặc baseline QuRA với ownership loss. Proposal transfer còn cấm
đọc victim fingerprint lúc xây dựng attack, nên quyền truy cập của pilot khác
protocol đó. Xem [phạm vi nghiên cứu](survey/Blind_Quantization_Stable_Signature.md).

Chạy trong GPU job bằng một lệnh; có thể chuẩn bị dependencies trước trên login
node nếu compute node không có mạng:

```bash
bash run_blind_quantization.sh
```

Launcher mặc định tự chọn/tạo `wmq` và cài dependencies còn thiếu bằng helper.
Ngay cả khi đang activate môi trường khác, nó vẫn chọn `wmq` qua `conda run`.
Đổi tên bằng `WMQ_CONDA_ENV`; dùng `WMQ_ENV_MODE=current` để cài/chạy ngay trong
Python hiện tại. Chế độ auto cũng dùng current khi không tìm thấy Conda.
Lần đầu launcher chuẩn bị fixture Stable Signature công khai tại
`output_artifacts/models/marked_sd21`; lần sau reuse. Bước này thuộc
vai trò organizer, nằm ngoài attacker. Với thí nghiệm cô lập, organizer chuẩn bị
fixture riêng rồi truyền `--model /path/to/marked_pipeline`.

Một lệnh trên tự chuẩn bị fixture và pool natural, chạy blind attack, tải/kiểm tra
owner extractor, rồi evaluate sau khi selection đã freeze. Key và extractor chỉ
được cấp cho process evaluator; `wmq_blind.py` không nhận hai dữ liệu này. Launcher
luôn dùng cùng Python đã cài và kiểm tra dependencies trong bước bootstrap.

SHA256 checkpoint Meta được so với pin trước `torch.load`; khóa liên tiến trình
và atomic mkdir ngăn các launcher ghi đè fixture. Completion marker được xuất cuối.
Backbone revision mặc định vẫn là `main`; dùng `prepare_marked_fixture.py --revision
<commit>` nếu cần pin toàn bộ backbone. Không coi pin decoder là pin toàn pipeline.

Mặc định: **W4, clip 1.0, toàn bộ weights decoder**, 32 train / 20 search / 100 test,
100 bước tối ưu, đánh giá search mỗi 10 bước. `prompt.txt` có 160 prompt viết tay;
đây chưa phải benchmark COCO/DrawBench. Các split có prompt và seed riêng.
Mặc định chỉ W4. Có thể truyền `--bits 4 3` để chủ động stress-test W3;
W8 được giữ trong code dự phòng và chỉ chạy khi truyền `--bits 8` tường minh.

Extractor official được pin SHA256
`77cd0a2040b9391233bbcd79c1adf00816b196089cbb844da40035f854637a04`
và kiểm tra lúc tải lẫn ngay trước `torch.jit.load`. Public key mặc định là
`111010110101000001010111010011010100010000100111`, đúng chuỗi trong ví dụ
đánh giá public SD2 decoder của repository Stable Signature chính thức. Evaluator
ghi nguồn/hash key và đặt `reference_validation.valid=false` nếu reference TPR
dưới 0,9; trong trường hợp đó vẫn lưu mọi số liệu nhưng không được kết luận attack.
Với `--model` riêng phải cấp `SS_KEY`; với extractor riêng phải cấp đồng thời
`SS_EXTRACTOR` và `SS_EXTRACTOR_SHA256`.

| Nhánh | Thay đổi quantizer |
|---|---|
| `fixed_ptq` | RTN cố định, không tune |
| `sensitivity` | Đo từng layer trên train, coordinate search scale theo thứ tự đo được |
| `rounding` | Học rounding từng weight theo pseudo-target làm mờ |
| `rounding_scale` | Học đồng thời rounding và scale per-channel có giới hạn |
| `reconstruction` | Học rounding với strength=0, đối chứng giữ ảnh tham chiếu |
| `natural_rounding` | Học rounding với MSE + LPIPS trên natural và bảo toàn ảnh sinh |
| `natural_rounding_scale` | Như natural_rounding, thêm per-channel scale |
| `natural_residual` | Học rounding với loss projection lên basis residual từ natural TRAIN; bảo toàn phần bù |
| `natural_qat_purification` | Học offset mã W4 nhiều ô với natural reconstruction; bảo toàn nội dung tần số thấp |
| `natural_residual_qat` | Cùng residual objective/basis, học code-offset với LR riêng và warm-up/cosine |
| `natural_full_finetune` | Đối chứng toàn bộ decoder FP32, không bị ràng buộc grid; đánh giá một lần |

Năm nhánh đầu là **model-only**, các nhánh natural là model + natural + LPIPS.
Mặc định focused khai báo `fixed_ptq`, `reconstruction`, `natural_rounding`,
`natural_residual`, `natural_residual_qat` và `natural_full_finetune`.
Các nhánh lượng tử hóa chỉ chạy W4; giữ FP32 upper bound. Các phương pháp khác vẫn
chạy được qua `--methods` / `--natural-methods`. Mọi nhánh dùng cùng bits, coverage, prompt/seed và
ngưỡng chất lượng trong một `comparison_group`. Scale được phép nằm trong
[0.8, 1.25] lần scale RTN khởi tạo cho các nhánh scale. Không tune bitwidth liên tục,
không sửa bias/norm/full-precision weights trong các nhánh quantization; đối chứng FP32
có comparison_group riêng và được phép sửa tất cả decoder parameters. Quantizer dùng đủ miền signed
`[-2^(b-1), 2^(b-1)-1]`, ví dụ W4 là `[-8, 7]`, với zero-point 0. UNet sinh latent FP16; **VAE và activation
của VAE chạy FP32**, nên pilot này không phải W4A16 của script fair cũ.

Profiler lượng tử hóa từng layer rồi đo thay đổi ảnh trên tối đa `--profile-n 16`
mẫu train. Nó đo MSE ảnh và mức cải thiện proxy, không đo ownership gradient.
Tất cả layer trong scope vẫn được lượng tử hóa; profiler điều khiển thứ tự search
scale, không âm thầm giảm coverage. `--scope late` vẫn là ablation heuristic riêng.
Profiler lưu số đo từng ảnh và khoảng bootstrap 95% cho priority, mặc định 1000
resample dùng cùng chỉ số ảnh giữa các layer (`--profile-bootstrap`). Khoảng này
đánh giá biến thiên trong pool train hiện có, không thay thế nhiều seed/model.
Với ít hơn 16 ảnh, script cảnh báo rồi vẫn chạy. Với một ảnh, CI để null. Thứ tự
layer vẫn dựa trên point estimate, không tuyên bố thứ hạng chắc chắn từ các CI này.

Lệnh mặc định đầy đủ là `bash run_blind_quantization.sh`. Ví dụ dưới đây là run
tùy chỉnh W4 trên model riêng, vì vậy phải cấp key đúng của model đó:

```bash
SS_KEY="<48-bit-key-cua-model>" bash run_blind_quantization.sh --direct \
  --model "$PWD/output_artifacts/models/marked_sd21" --prompts "$PWD/prompt.txt" \
  --output "$PWD/output_attack/blind_w4_seed3407" \
  --bits 4 --clips 1.0 --steps 100 --eval-every 10 --strength 0.25
```

Chạy smoke trước (cần sáu prompt khác nhau):

```bash
bash run_blind_quantization.sh --direct --train-n 2 --search-n 2 --test-n 2 \
  --bits 4 --clips 1.0 --steps 2 --eval-every 1 --profile-n 1
```

Reconstruction control đã có mặc định. Dùng `--methods` để giới hạn các nhánh
model-only; `--natural-methods` chọn nhánh natural khi có natural dataset.
Mỗi nhánh khởi tạo độc lập. Random rounding đã bỏ khỏi code;
các nhánh gradient dùng `steps` update, sensitivity dùng `steps` proposal train.
RTN được đánh giá một lần. Các nhánh tối ưu có RTN fallback được gắn nhãn rõ.
Đây là so sánh cùng budget lượng tử hóa/chất lượng, **chưa cùng chi phí tính toán**;
report ghi số train/search evaluation, valid gradient update và proposal bị từ chối.
`selection.json` có thêm số image forward, backward attempt và các biến thực sự
được tối ưu. `report.json/branch_compute` ghi wall time search/export (gồm profiler
nếu có), không phải latency inference INT4 và không bao gồm test. Không coi một
forward cả tập train là tương đương một gradient step một ảnh.

Nếu truyền nhiều `--bits`/`--clips`, mỗi tổ hợp có đầy đủ các nhánh và lựa chọn riêng;
không chọn một winner toàn cục rồi so W2 với W4. Không dùng owner metrics để chọn lại
strength, bitwidth, clip hoặc seed của cùng blind experiment.

Ngưỡng quality: mean PSNR >=25, mean SSIM >=0.9, min per-image PSNR >=22 dB, so với
ảnh marked gốc. Mặc định report-only không lọc candidate: chọn theo objective,
ghi `selected_quality_failed` và `search_feasible=false` nếu quality trượt.
Chế độ constrained giữ fallback `no_feasible_candidate` khi không có candidate đạt.
Không chế độ nào dừng vì quality hữu hạn thấp. Nếu test trượt, report
ghi `quality_failures`. Chúng không được tính là attack thành công. Ngưỡng này
không tương đương LPIPS gate trong script fair cũ.

SSIM hiện dùng Gaussian **11×11, sigma=1.5**, population covariance, loại biên
5 pixel và trung bình kênh RGB; thống kê tính FP64, VAE vẫn chạy FP32. Đã đối chiếu
với `skimage.metrics.structural_similarity`. Định nghĩa được ghi vào manifest;
không gộp kết quả SSIM 5×5 cũ với run mới như cùng một metric. Gaussian 5×5 trong
pseudo-target/loss làm mờ vẫn giữ nguyên. Xem [tài liệu scikit-image](https://scikit-image.org/docs/stable/api/skimage.metrics.html).

Mỗi lần chạy tạo thư mục mới `output_attack/run_<YYYYMMDD_HHMMSS>` và file log:

```text
output_attack/blind_w4_seed3407/
  manifest.json                 # Protocol bất biến + danh sách test_branches
  profile_w4_c1.0_all.json       # Đo từng layer trên train
  search.json                   # Toàn bộ candidate, không có owner score
  selection_frozen.json         # Hash manifest, search, profiler, mọi nhánh trước test
  report.json                   # Selection/quality riêng từng nhánh
  quality_warnings.jsonl        # Cảnh báo metric/update, ghi ngay khi xảy ra
  branches/<method>_w4_c1.0/
    search.json
    updates.json                # Nhánh có update/proposal
    selection.json
output_artifacts/checkpoints/blind_w4_seed3407/
  branches/<method>_w4_c1.0/
    quantizer.json
    quantizer.safetensors       # Integer codes + scale, kiểm tra khớp exported weights
    vae/                        # Diffusers VAE đã dequantize FP32
output_artifacts/images/blind_w4_seed3407/
  marked_reference_test/
  pseudo_target_test/
  <method>_w4_c1.0_test/
```

Tất cả nhánh được khóa trước khi sinh latent test đầu tiên. Test chỉ đánh giá,
không chọn lại. Export là simulated PTQ; lưu integer codes không đồng nghĩa đã có
packed INT4 kernel hoặc chứng nhận tương thích một backend triển khai.

Sau khi attack khóa selection và tạo ảnh test, launcher **tự gọi** owner evaluator
với public fixture mặc định. Lệnh bên dưới chỉ dùng khi cần đánh giá thủ công một
run đã có:

```bash
python src/evaluate_blind_watermark.py \
  --run "$PWD/output_attack/blind_w4_seed3407" \
  --extractor /path/to/dec_48b_whit.torchscript.pt \
  --expected-extractor-sha256 77cd0a2040b9391233bbcd79c1adf00816b196089cbb844da40035f854637a04 \
  --key 111010110101000001010111010011010100010000100111 \
  --key-source "facebookresearch/stable_signature public SD2 decoder example key" \
  --fpr 0.001 --lpips
```

Key trên thuộc fixture Stable Signature công khai. Extractor chính thức:
[Meta](https://dl.fbaipublicfiles.com/ssl_watermarking/dec_48b_whit.torchscript.pt).
Không cấp chúng cho attacker. Evaluator đọc nhánh động từ manifest, kiểm tra hash
của tất cả checkpoint/quantizer/search/profiler và ảnh test, rồi ghi
`owner_evaluation.json`/`watermark_retention.csv`. Reference được xác định bằng
`reference_label`, không dựa vào thứ tự. Mỗi dòng có comparison group và quality
flags, bit accuracy, TPR, retention, survival, LPIPS nếu bật.
TPR và survival có Wilson CI 95%; mức giảm TPR có paired-image bootstrap CI 95%
(2000 resample, seed 3407), báo theo **điểm phần trăm**. Ví dụ 50/100 ảnh được phát
hiện có Wilson CI khoảng [40.4%, 59.6%], không phải ±6%. Các CI là pointwise,
giả định ảnh độc lập; không bao gồm biến thiên giữa model/key/seed và không chứng
minh 100 prompt viết tay đại diện deployment. Khoảng bootstrap có thể co về một
điểm khi mọi cặp có cùng outcome; Wilson vẫn thể hiện bất định ở TPR=0/1.
Nguồn: [SciPy proportion_ci](https://docs.scipy.org/doc/scipy-1.16.1/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).

FPR dùng giả định bit ngẫu nhiên độc lập, chưa phải empirical FPR. Threshold vượt
độ dài key bị từ chối. Reference không được phát hiện đầy đủ sẽ có cảnh báo và
`baseline_detected_count`; không diễn giải TPR thấp vốn có thành thành công attack.
Metadata/hash phát hiện artifact thay đổi, không chống người cố ý sửa đồng thời
source và tất cả metadata. Run cũ schema 1 vẫn được evaluator hỗ trợ.

Nạp một VAE cụ thể để inference:

```python
import torch
from diffusers import AutoencoderKL, StableDiffusionPipeline, DDIMScheduler

pipe = StableDiffusionPipeline.from_pretrained(
    "output_artifacts/models/marked_sd21", torch_dtype=torch.float16)
pipe.vae = AutoencoderKL.from_pretrained(
    "output_artifacts/checkpoints/blind_w4_seed3407/branches/rounding_scale_w4_c1.0/vae",
    torch_dtype=torch.float32)
pipe.to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
with torch.inference_mode():
    z = pipe("a lighthouse above a stormy sea", num_inference_steps=25,
             guidance_scale=7.0, output_type="latent").images
    x = pipe.vae.decode(z.float() / pipe.vae.config.scaling_factor, return_dict=False)[0]
    pipe.image_processor.postprocess(x, output_type="pil")[0].save("sample.png")
```

Kiểm thử:

```bash
bash -n run_blind_quantization.sh
python -m pip install -r requirements-test.txt  # chỉ cần cho test đối chiếu SSIM
python -m unittest discover -s tests -v
```

Test CPU dùng pipeline nhỏ và VAE Diffusers thật: thay riêng test không đổi bất kỳ
branch selection/export nào; không gọi selection sau khi sinh test; rollback NaN
không được đếm là update hợp lệ; scale/codes hợp lệ và reload VAE khớp. Các test
không chứng minh hiệu quả xóa watermark, VRAM hoặc runtime của full checkpoint.
Manifest ghi GPU, PyTorch/CUDA/cuDNN, Python, TF32 và deterministic settings;
owner evaluation cũng ghi runtime. Cùng seed và tắt TF32 không bảo đảm kết quả
bitwise giống nhau giữa các GPU/thư viện. VAE Stable Signature không có trục
timestep; các kết quả ở đây không áp dụng thành kết luận về carrier UNet/latent.
# Owner diagnostics bổ sung, không checkpoint/resume

Lệnh mặc định vẫn là `bash run_blind_quantization.sh`. Không thêm checkpoint tiến
trình, optimizer state hay resume. Artifact quantizer/VAE cuối mỗi nhánh vẫn được
giữ để đánh giá và kiểm tra hash như trước.

Ảnh RGB tại các biên dữ liệu phải là FP32 NCHW, hữu hạn và trong [0,1]. Output
VAE được kiểm tra hữu hạn trước clipping; raw output phục vụ training được phép
vượt [0,1]. Các kiểm tra này không tự ép dtype để che lỗi preprocessing.

`watermark_retention.csv` bổ sung tổng correct/error bits, net additional bit
errors, số ảnh decoding improved/worsened/tied, exact-message accuracy, và mức
giảm bit accuracy kèm paired bootstrap CI95 theo ảnh. “Decoding worsened” nghĩa
là ít bit khớp hơn; không đồng nghĩa evasion tốt hơn vì detector dùng double-tail.

Sau khi mọi nhánh đã freeze và owner metrics đã xuất, launcher chạy phân tích
gradient trên 4 mẫu đầu của test set, dùng lại prompt/seed đã khai báo:

```bash
WMQ_MECHANISM_SAMPLES=16 bash run_blind_quantization.sh
# 0 tắt riêng phân tích gradient; vẫn chạy owner evaluation đầy đủ.
```

`mechanism_analysis.csv` chứa từng method/sample/layer: norm quantization error,
norm gradient, dot product và cosine giữa error và gradient ownership/quality.
`mechanism_analysis.json` ghi định nghĩa loss, hash freeze/extractor và giới hạn.
Không lưu thêm model hoặc latent cho diagnostic này. Nó tái sinh latent, giải phóng
UNet khỏi GPU rồi backprop từng mẫu qua VAE/extractor, nên có thêm thời gian/VRAM.

Gradient được tính tại **quantized endpoint**. Quality loss là RGB MSE so với
marked decoder cùng latent. Ownership score là bình phương trung bình signed
soft agreement `tanh(logit/2)*(2*key-1)`; cả đảo toàn bộ bit lẫn khớp toàn bộ bit
đều có score cao. Score này là proxy khả vi, không thay detector/FPR. Dot ownership
âm biểu thị xu hướng giảm score cục bộ; không chứng minh hiệu ứng hữu hạn hoặc
ownership subspace. Cosine của vector bằng 0 được ghi null.

Chỉ owner evaluator có key/extractor; attacker không import module diagnostic.
Artifact được kiểm tra hash trước và sau phân tích; không chọn lại candidate từ
kết quả này. Latent tái sinh có thể lệch số học do CUDA/OOM/batch, và gradient dùng
ảnh float trước làm tròn PNG. Bốn mẫu là diagnostic pilot, không đại diện toàn bộ
test set. Các CI theo ảnh là pointwise, có điều kiện theo model/key, không phải
CI đồng thời cho nhiều phương pháp.


## Experimental watermark embedded in diffusion UNet weights

`bash run_new.sh` runs the new teacher training, merged-model export, LoRA transfer
pilots, fresh rank-16/32 students and final evaluation. Native teacher strength is
required (default TPR >=95%, Wilson lower bound >=90%); failure is reported without
mislabeling weak-watermark inheritance as success. This is an unproven research
method, with CPU integration validation but no completed full GPU efficacy result.
See [new_watermark/README.md](new_watermark/README.md) for the exact protocol, runtime
requirements, meta-learning approximation, controls and result paths.
