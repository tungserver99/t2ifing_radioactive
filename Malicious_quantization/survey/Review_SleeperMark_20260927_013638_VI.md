# SleeperMark: review run 20260927_013638_836714

Nguồn: `output_attack/sleepermark_20260927_013638_836714/`, các file
`watermark_retention.csv`, `fid.csv`, `ordinary_fid.csv`, `*_training.csv`, `manifest.json`.
100 ảnh TEST, baseline TPR 100%, BA 99.583%, detector double-tail 48 bit,
threshold inclusive ≥36 hoặc ≤12, FPR tổng mục tiêu 0.001.

## Kết quả

| Nhánh | BA | TPR ↓ | SSIM ordinary | SSIM triggered | FID triggered ↓ | Joint success |
|---|---:|---:|---:|---:|---:|---:|
| Fixed PTQ W4 | 99.333% | 100% | 0.6820 | 0.6792 | 101.04 | 0/100 |
| CFG reconstruction W4 | 98.563% | 100% | 0.7354 | 0.7187 | 88.68 | 0/100 |
| Equivariance W4 | 98.396% | 99% | 0.7328 | 0.7200 | 87.86 | 0/100 |
| FP32 base + delta4 | 99.375% | 100% | 0.9078 | 0.8970 | 33.13 | 0/100 |

Delta cải thiện chất lượng rõ rệt so với các nhánh W4 nhưng chưa làm yếu
watermark đáng kể. Không gọi delta tốt hơn về attack: nó có base FP32 và
ngân sách biểu diễn khác W4. TPR 99% của equivariance chỉ tương ứng một ảnh
thoát detector, ảnh đó không đạt ngưỡng chất lượng; joint success vẫn bằng 0.
FID đo với marked reference và chỉ 100 ảnh, không tương đương FID benchmark
trên tập real lớn. SSIM trung bình tốt không bảo đảm mọi ảnh đều đạt chất lượng:
delta chỉ có 65/100 ảnh triggered đồng thời đạt SSIM≥0.8 và PSNR≥25.

Thời gian TRAIN được ghi ở bước 2000: CFG khoảng 17.43 phút, equivariance
27.46 phút, delta 34.85 phút; chưa bao gồm chuẩn bị, sinh ảnh TEST, detector và FID.
Peak allocated VRAM tương ứng 7.33, 7.33, 8.16 GiB trong run này; không phải
tổng bộ nhớ thiết bị hay cam kết cho cấu hình khác.

Delta có 24.246% code khác 0 ở cuối, RMS delta khoảng 0.00013576.
Nó không bị đứng yên. Giới hạn chính là objective: học sửa spatial defect trên
ordinary prompts không mặc nhiên làm thay đổi hành vi khi có trigger.
[SleeperMark](https://arxiv.org/html/2412.04852v2) chủ động tách hành vi watermark
khỏi ngữ nghĩa thông thường. Kết quả hiện tại phù hợp với giới hạn này, nhưng
chưa đủ để chứng minh nguyên nhân duy nhất.

## Cải tiến đã triển khai

| Nhánh đang bật | Thay đổi | Mục đích / giới hạn |
|---|---|---|
| Fixed PTQ | Chọn scale theo MSE trọng số từng group, dùng đủ signed range và thử clipping | Giảm sai số khởi tạo; vẫn là đối chứng PTQ benign |
| CFG reconstruction | Khởi tạo mới; LR code 0.01, scale 0.001; warmup/cosine; timestep stratification | Cho mã integer cập nhật hiệu quả hơn và phủ đều timestep; không tự tạo mục tiêu xóa watermark |
| Equivariance | Các sửa đổi W4; spatial teacher/student đều dùng CFG; chuẩn hóa loss theo timestep; thử prompt có prefix ngẫu nhiên | Đồng nhất với inference và mở rộng calibration, chưa chứng minh tìm được hành vi watermark |
| Delta CFG equivariance | Optimizer/schedule và timestep coverage mới; spatial normalization, context augmentation; giữ radius 0.05 | Sửa mục tiêu học thay vì tăng biên độ delta hoặc số bước một cách thiếu bằng chứng |

Khởi tạo MSE gồm chính RTN cũ trong tập ứng viên. Vì vậy sai số trọng số không
tăng theo tiêu chí này; chất lượng ảnh sau toàn bộ diffusion vẫn cần đo lại.
Group size giữ 64 cho W4, delta vẫn per-channel. Không thêm fine-tune FP32.

Optimizer code/scale có đơn vị và độ lớn gradient khác nhau. Balanced mode tách
LR, dùng epsilon 1e-12 và clamp tham số sau update để tránh vùng gradient bằng 0
ở ngoài giới hạn. CSV báo code-change fraction và LR để phát hiện optimizer
không vượt ô integer. Không suy luận hiệu quả watermark từ số code đã đổi.

Timestep sampler đi qua các mốc đã cache theo thứ tự xáo trộn, rồi lấy mẫu TRAIN
trong mỗi mốc; auxiliary late-time có sampler riêng. Cách này giảm rủi ro bỏ sót
mốc trong ngân sách hữu hạn. [Q-Diffusion](https://arxiv.org/abs/2302.04304) cũng
nhấn mạnh calibration theo timestep; implementation ở đây không tái hiện đầy đủ
thuật toán của paper.

Với epsilon prediction, đạo hàm của x0 theo noise prediction có bình phương
`(1-alpha)/alpha`. Spatial MSE mới chia theo hệ số này (floor 0.01), rồi chia
`7.5²` để cùng đơn vị với CFG reconstruction. Có xử lý riêng v-prediction và
sample prediction. Đây là chuẩn hóa theo scheduler, không phải ownership weighting.

Một nửa auxiliary TRAIN batch dùng prefix punctuation ngẫu nhiên, độc lập với
owner assets. RNG context tách khỏi RNG timestep và shift để dễ ablate. Latent
vẫn từ ordinary trajectory: đây là perturbation calibration, chưa phải rollout
đúng theo prompt biến thể. Không có kết luận đã khôi phục trigger. Chất lượng
và detector vẫn chỉ được đánh giá sau khi đóng băng checkpoint.

## Chạy và đối chiếu

Lệnh chung vẫn chạy hai watermark:

```bash
bash run_blind_quantization.sh
```

Chỉ chạy SleeperMark:

```bash
bash run_blind_quantization.sh --watermark sleepermark
```

Tái lập các lựa chọn thuật toán trước đợt sửa này:

```bash
bash run_blind_quantization.sh --watermark sleepermark --weight-init rtn --quant-refinement legacy
```

Ablate từng thay đổi bằng `--weight-init rtn`, `--spatial-loss-mode x0`,
`--spatial-context-probability 0`. Mặc định giữ 2000 bước và bốn nhánh cũ;
không thêm sweep, seed hay nhánh fine-tune. Thời gian equivariance có thể tăng
do thêm unconditional passes; activation checkpointing vẫn bật.

Thêm `*_initialization.csv` ghi MSE từng layer trước/sau chọn scale. Manifest
ghi cấu hình và hash code. Các kiểm thử CPU gồm MSE không tăng, giới hạn grid,
phủ timestep, công thức loss/gradient với ba prediction types và train/export
UNet diffusers nhỏ. Cần run GPU mới để biết BA/TPR/chất lượng có cải thiện không.
Các kết quả đã đọc là development evidence; đánh giá xác nhận cần prompt/key
chưa dùng để điều chỉnh phương pháp.
