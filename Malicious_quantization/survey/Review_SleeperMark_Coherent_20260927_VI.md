# SleeperMark: kết quả hoàn chỉnh và thử mục tiêu residual chung giữa nhiều ảnh

## Kết quả thực tế

Đã tải JSON/CSV từ server của hai run:
`sleepermark_20260927_061035_583957` và `sleepermark_20260927_075826_683759`.
Cùng baseline BA 99.5833%, TPR 100/100, detector double-tail ≥36 hoặc ≤12 trên
48 bit; FPR tổng mục tiêu 0.001. Mỗi nhánh 100 ảnh triggered + 100 ordinary.

| Phương pháp | BA ↓ về 50% | TPR ↓ | SSIM triggered | PSNR triggered | Joint success |
|---|---:|---:|---:|---:|---:|
| Fixed PTQ W4 | 99.04% | 100% | 0.688 | 19.90 | 0/100 |
| CFG reconstruction W4 | 94.94% | 98% | 0.621 | 18.16 | 0/100 |
| Equivariance W4 | 94.21% | 98% | 0.631 | 18.45 | 0/100 |
| Conditional rollout W4 | 94.23% | 98% | 0.632 | 18.49 | 0/100 |
| Marked FP32 base + delta4 | 99.23% | 100% | 0.959 | 33.28 | 0/100 |

Joint success đòi hỏi baseline detected, output thoát double-tail detector và
đồng thời SSIM≥0.8, PSNR≥25 trên chính ảnh triggered. Hai ảnh thoát detector của
các nhánh W4 không qua chất lượng. Delta có 93/100 ảnh qua chất lượng nhưng không
ảnh nào thoát detector. Delta có triggered FID-to-marked 13.93; CFG/equivariance
lần lượt 116.85/115.16. FID chỉ 100 ảnh, dùng marked reference, không tương đương
FID real benchmark. Không dùng metric owner để chọn checkpoint trong run.

So với run cũ 013638, W4 học giảm BA nhiều hơn nhưng chất lượng kém hơn. Rollout
không cải thiện đáng kể. Delta cải thiện chất lượng chứ chưa cải thiện attack.
Đây chưa phải kết quả đủ mạnh cho claim watermark removal, càng chưa chứng minh
tính chọn lọc ownership hoặc novelty của một bài phương pháp.

Tắt equivariance W4 và conditional rollout khỏi mặc định, vẫn giữ implementation
để đối chứng. Giữ Fixed PTQ, CFG reconstruction và delta spatial làm controls.

## Vì sao thay mục tiêu

Đọc lại [SleeperMark, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_SleeperMark_Towards_Robust_Watermark_against_Fine-Tuning_Text-to-image_Diffusion_Models_CVPR_2025_paper.pdf):
watermark được gắn với trigger văn bản và chủ động tách khỏi hành vi ordinary.
Calibrate trên ordinary prompts có thể không kích hoạt đường watermark. Phá
equivariance chung cũng có thể chỉ phá texture, đúng với chất lượng quan sát.

[SKD-CAG](https://arxiv.org/html/2508.18235v1) sử dụng model bị can thiệp làm teacher
trên clean prompt để sửa hành vi với trigger. Nhưng threat model của họ biết trigger
(hoặc một phần trigger). [Spatial Attention Unlearning](https://arxiv.org/html/2504.18563v1)
cũng giả định biết trigger. Không được mượn con số removal của các bài này làm
cam kết cho thí nghiệm blind quantizer-only.

Hướng thử mới đặt câu hỏi: một perturbation embedding nhỏ dùng chung cho nhiều
prompt độc lập có kích hoạt một residual lặp lại giữa các ảnh không? Nếu có,
quantizer có thể học giảm thành phần chung ấy với ít mất chất lượng hơn không?
Đây là giả thuyết, không phải kết luận residual chung chính là ownership.

## Implementation

Hai nhánh `coherent_probe_qat` (W4 group 64) và `delta_coherent_probe`
(immutable marked FP32 base + delta4). Nhánh delta không phải full-model W4.
Chỉ học quantizer; không fine-tune tự do hoặc tải clean reference model.

1. Chọn ít nhất hai prompt TRAIN khác nhau tại cùng timestep thấp. Lấy latent từ
   trajectory ordinary, không dùng ảnh TEST hay trigger của owner.
2. Một perturbation liên tục `delta_c` dùng chung cho các context, không thay BOS,
   chỉ tác động bốn token đầu sau BOS. L2 radius ≤15% norm các token đó của từng
   context. Giữ probe qua nhiều minibatch để thử tính ổn định qua nội dung.
3. Với response `r_i = guidance * (x0(c_i+delta_c)-x0(c_i))`, tính high-pass `h_i`.
   Mục tiêu probe là trung bình inner product giữa **các ảnh khác nhau** trừ
   low-pass energy. Loại self-energy để không nhầm residual lớn của một ảnh với
   tín hiệu chung. Dùng autograd chỉ theo context trên marked teacher đã freeze.
4. Chỉ giữ iterate probe tốt nhất theo objective TRAIN trong các bước PGD, không
   dùng owner. Chuẩn hóa agreement bằng mean high-pass energy, clamp về [0,1],
   nhân với mean high-pass response. Agreement âm cho correction bằng 0.
5. Target là CFG predicted x0 của teacher ở probed context trừ phần chung vừa tìm.
   Giữ border hai pixel, cap correction RMS 0.05. Quantizer học target đó cùng
   ordinary CFG reconstruction loss, vẫn bao phủ toàn miền timestep.
6. Export checkpoint cuối bước 2000, freeze rồi owner evaluate BA/TPR/quality/FID.

Probe được cập nhật hai bước gradient mỗi bốn optimizer steps. Ngoài những lần đó,
probe được tái dùng và response được đo lại trên minibatch mới. Mọi context/target
được detach trước khi backward student, tránh teacher nhận gradient quantizer.
W4 mới dùng code LR 0.003, scale LR 0.0001, thấp hơn balanced cũ đã làm mất chất lượng.
Delta mới giữ code LR 0.01 và dùng scale LR 0.0001.

Hạn chế: continuous context có thể không tương ứng token hợp lệ; tín hiệu chung
có thể là bias/texture bình thường. Hai prompt là estimator nhỏ; cần ablation
batch lớn hơn và kiểm tra transfer sang prompt/key chưa thấy. Không khẳng định
đã phục hồi trigger hoặc nhận diện được ownership subspace.

## Thí nghiệm tiếp theo

Giữ seed 3407, 2000 steps, cùng TRAIN/TEST. Chạy song song hai nhánh mới cùng hai
control với LR giảm tương ứng, để tránh gán cải thiện chất lượng do LR cho objective:

```bash
bash run_blind_quantization.sh --watermark sleepermark \
  --methods cfg_reconstruction delta_cfg_equivariance coherent_probe_qat delta_coherent_probe \
  --code-lr .003 --lr .0001
```

Control học trên ordinary batch 1, hai nhánh mới thêm auxiliary batch 2; không
gọi đây là so sánh cùng FLOPs. Báo runtime và logical forward count, sau đó cần
control cùng compute budget. `--probe-steps 0` là shared-random-context ablation.
`--coherent-weight 0` tắt toàn bộ auxiliary objective để kiểm tra LR/optimizer.

Log có `coherent_agreement`, `coherent_pair_energy`, `coherent_lowpass_energy`,
`coherent_correction_rms`, score trước/sau probe, code changes và ordinary loss.
Nếu agreement/correction gần 0 hoặc owner TPR vẫn 100%, phải kết luận probe chưa
tìm được tín hiệu hữu ích. Không tăng ngân sách vô hạn hoặc tuyên bố đột phá từ
training loss. Những run đã dùng để chỉnh phương pháp là development, cần một
evaluation mới được khai báo trước để xác nhận kết quả cuối bài.

Các nhánh tiếp tục dùng process riêng; mặc định ước lượng 18 GiB/process để dành
headroom cho gradient theo context batch 2. Concurrency là ước lượng dung lượng,
không phải phép đo SM occupancy. `nvidia-smi` utilization 100% chỉ biểu thị GPU
đang chạy kernel gần như suốt khoảng đo; không chứng minh đạt 100% compute tối đa.

## Kiểm tra implementation

Test CPU gồm UNet Diffusers thật, gradient checkpointing, delta export/reload,
khôi phục hash nguồn, giới hạn context perturbation, chọn prompt độc lập, pairwise
score loại self-energy và correction bằng 0 khi response ngược chiều. Các test
đều pass. Bash syntax cũng pass.

Smoke test CUDA đã chạy cả hai nhánh trên checkpoint SleeperMark SD1.4 đầy đủ,
hai optimizer steps với batch probe 2: peak allocated 8.16 GiB (W4) và 8.54 GiB
(delta), output trọng số hữu hạn và source hash được khôi phục. Smoke dùng latent
ngẫu nhiên để kiểm tra thực thi, không phải thí nghiệm attack. Ở minibatch smoke,
agreement âm nên correction bằng 0 đúng thiết kế; delta codes chưa đổi sau hai
bước. Không dùng smoke này làm bằng chứng suppression.

Bảng `parallel_summary.csv` bổ sung baseline BA/TPR, FID triggered/ordinary,
trạng thái FID và thời gian/forward count/peak VRAM TRAIN khi có log. FID thiếu
hoặc lỗi được báo rõ, không suy diễn từ trạng thái owner evaluation.
