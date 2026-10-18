# SleeperMark: survey và thí nghiệm spatial QAT

Ngày khảo sát: 26/09/2026. Đây là thiết kế thử nghiệm mới, chưa có kết quả attack trên checkpoint SleeperMark thật.

## 1. Tại sao đổi hướng?

Run `sleepermark_20260926_052518_273157`: mọi nhánh TPR 100/100; BA thấp nhất 97.60%, trong khi SSIM ảnh thường W4 khoảng .58–.63. Không có bằng chứng rằng natural denoising hiện tại chạm được vào cơ chế watermark. Tiếp tục tăng bước theo cùng loss không có cơ sở đủ mạnh.

SleeperMark nhúng residual latent cố định, không phụ thuộc ảnh; dùng trigger kích hoạt và điều chỉnh mạnh hơn ở timestep thấp. Appendix A giải thích việc tách chức năng watermark khỏi ngữ nghĩa, Appendix C mô tả lịch học theo timestep. Điều này gợi ý một phép thử dựa trên vị trí không gian và thời gian, nhưng **không chứng minh watermark sẽ bị xóa bởi phép thử đó**. [Paper, §4.1–4.2 và Appendix A/C](https://arxiv.org/html/2412.04852v2).

## 2. Related work và giới hạn áp dụng

| Công trình | Điều đáng học | Giới hạn so với threat model hiện tại |
|---|---|---|
| [SleeperMark, CVPR 2025](https://arxiv.org/html/2412.04852v2) | Residual cố định, cơ chế kích hoạt và ưu tiên timestep thấp | Không thể lấy residual/trigger bí mật làm target rồi gọi là blind |
| [SKD-CAG, 2025](https://arxiv.org/html/2508.18235v1) | Dùng phản hồi prompt thường của chính model làm teacher; loss prediction và cross-attention | Giả định biết trigger hoặc một phần trigger; kết quả trên backdoor pixel/style không phải chứng minh chống SleeperMark |
| [T2IShield, 2024](https://arxiv.org/html/2407.04215v2) | Phân tích cross-attention để phát hiện, định vị rồi chỉnh sửa trigger | Cần prompt nghi vấn; detector được hiệu chỉnh bằng benign/backdoor examples. Không giải quyết tự động bài toán chỉ có checkpoint và chưa biết trigger |
| [TNC-Defense, bản v2 năm 2026](https://arxiv.org/html/2602.01765v2) | Phát hiện bất thường theo timestep và sửa quỹ đạo ở khoảng liên quan | Phần sửa dùng prompt bất thường và ảnh từ một reference model khác; “trigger-agnostic” không đồng nghĩa model-only của ta |
| [Textual Perturbations Defense, ECCV workshop 2024](https://arxiv.org/abs/2408.15721) | Perturbation của conditioning có thể ảnh hưởng backdoor | Sửa input khi inference khác với thay trọng số UNet; không dùng nó thay attack quantizer |
| [Q-Diffusion, ICCV 2023](https://arxiv.org/abs/2302.04304) | Calibration theo timestep quan trọng cho PTQ diffusion | Bảo toàn chất lượng, không có mục tiêu xóa watermark |

Không lấy tỷ lệ thành công của các paper trên backdoor khác làm kỳ vọng định lượng cho SleeperMark. Không tuyên bố “đầu tiên”: cần khảo sát tiếp và thực nghiệm trước khi chốt novelty.

## 3. Giả thuyết mới: kiểm tra thành phần cố định theo tọa độ

Gọi `f(z,t,c)` là dự đoán x0 của UNet marked, `S` là phép dịch latent. Xét:

\[
 A f(z,t,c)=\tfrac12[S^{-1}f(Sz,t,c)+S f(S^{-1}z,t,c)].
\]

Trong mô hình lý tưởng `f=g+a(c)δ`, nếu phần nội dung `g` equivariant với phép dịch thì `A` giữ `g` và đổi residual cố định `δ` thành trung bình các bản dịch ngược. Các thành phần Fourier không bất biến với phép dịch có thể suy giảm; thành phần DC không suy giảm. Hai phép dịch không bảo đảm triệt tiêu mọi tần số, và có thể đảo dấu một số thành phần.

**Đây là lập luận dưới giả định, không phải định lý về UNet thật.** Padding, attention, vị trí vật thể và biên ảnh đều có thể gây thiếu equivariance dù không có watermark. Vì vậy code:

- Dùng hai dịch ngược chiều với độ dịch ngẫu nhiên 1–4 ô latent mỗi trục.
- Loại vùng biên khỏi correction để hạn chế ảnh hưởng seam của `roll`; điều này không loại hoàn toàn ảnh hưởng seam qua receptive field/attention.
- Chỉ lấy high-pass của chênh lệch ensemble so với teacher gốc.
- Giới hạn RMS correction mỗi ảnh ở .05 đơn vị latent x0.
- Không biến đổi prompt hay ảnh ở inference; tất cả biến đổi chỉ dùng để tạo target calibration.

Target:

\[
 y=f+\operatorname{cap}_{.05}\{M\,H(Af-f)\},\qquad
 L=L_{CFG}+\lambda\|\hat x_{0,Q_\psi}(z_t,t,c)-y\|^2.
\]

`H` là high-pass cố định, `M` loại biên. `y` vẫn là **pseudo-target**, không được gọi là ảnh/latent sạch. Bản thân `H` có thể làm thay đổi texture tự nhiên.

## 4. Hai nhánh đã triển khai

### `equivariance_qat`

Học mã và scale W4 trong UNet từ target không gian trên. Loss CFG/unconditional bảo toàn phản hồi ordinary ở toàn trajectory. Auxiliary loss dùng các điểm có t ≤ .35T. Không tải model teacher sạch; teacher là chính trọng số marked bất biến khi tắt parametrization.

### `adversarial_equivariance_qat`

Giữ loss trên, nhưng ở một phần bước huấn luyện tìm conditioning làm tăng spatial defect của teacher:

\[
 \max_{\|u\|_2\leq\rho\|c_{1:k}\|_2}
 \|MH(S^{-1}f(Sz,t,c+u)-f(z,t,c+u))\|^2.
\]

Mặc định perturb 4 context token đầu sau BOS, radius .15, 2 bước gradient ascent mỗi 4 update sau warmup. Sau đó stop-gradient context, tạo target bằng teacher tại **cùng conditioning đã perturb**, rồi backward vào quantizer. Không ép mọi prompt biến dạng về prompt gốc vì có thể xóa ngữ nghĩa thật.

Đây là continuous embedding probe, **không phải khôi phục trigger rời rạc**. Không có đảm bảo probe kích hoạt cùng đường watermark với trigger của owner. Phương pháp có thể thất bại vì chỉ tìm được độ nhạy ngữ nghĩa hoặc tác động của biên. Các token/embedding của owner không được đọc trong search.

## 5. Calibration và đối chứng

Mặc định lấy 8 điểm **trước UNet forward** trên mỗi trajectory DDIM sinh ảnh ordinary TRAIN. Không cộng noise lại vào latent cuối để giả làm trajectory. Cache CPU lưu sample và timestep thực; condition/prompt được ghi nhận. Điều này tăng số record trên cùng số ảnh calibration, không tăng số prompt độc lập.

Mặc định chạy:

1. `fixed_ptq`: RTN W4 group64.
2. `cfg_reconstruction`: cùng quantizer, calibration và loss bảo toàn, không spatial loss.
3. `equivariance_qat`: thêm spatial target, không adversarial probe.
4. `adversarial_equivariance_qat`: thêm continuous context search.

`prefix_consistency_qat` và các nhánh natural cũ vẫn có qua `--methods`, không chạy mặc định. Nhánh prefix trước đó chưa có kết quả; bỏ khỏi mặc định vì cơ sở cơ chế yếu hơn, không phải vì đã đo và chứng minh vô dụng.

Những thay đổi này vẫn chỉ tác động các weight ma trận thuộc scope UNet đã chọn. VAE/text encoder và các tham số ngoài scope giữ nguyên. W4 là dequantized FP32, chưa có kernel INT4. Không phải fine-tune FP32 rồi gắn nhãn quantizer-only.

## 6. Đánh giá và diễn giải

Giữ detector double-tail, đóng băng checkpoint trước owner evaluation. Thêm PSNR/SSIM của **chính ảnh trigger** và `joint_success_rate`: số ảnh vừa né detector, vừa đạt SSIM≥.8 và PSNR≥25, chia cho số ảnh baseline có watermark được phát hiện. Ngưỡng chỉ để báo cáo, không chặn chạy/chọn checkpoint.

`watermark_retention.csv` chứa tổng hợp; `<label>_triggered_quality.csv` chứa từng ảnh. FID trigger/ordinary vẫn tách riêng. FID100 so với marked chỉ là chỉ báo thăm dò, không phải FID trên benchmark lớn.

Log mới: `ordinary_cfg_loss`, `auxiliary_loss`, `teacher_spatial_defect`, `correction_rms`, `probe_initial`, `probe_final`, VRAM, `elapsed_seconds`, `logical_unet_forwards`. Số forward này không đếm lại từng block khi gradient checkpointing recompute. Probe metrics chỉ xuất ở bước có probe được log; tăng probe score không chứng minh watermark yếu đi.

Đối chứng cần làm trước khi claim phương pháp:

- Spatial weight=0 so với CFG: kiểm tra overhead/optimizer không gây lợi thế giả.
- Probe radius=0, probe steps=0 (random context cùng radius), và PGD đầy đủ. So sánh thêm compute-matched, vì PGD tốn nhiều forward hơn.
- Cùng chất lượng ordinary **và triggered**, so TPR/joint success; không chỉ BA giảm.
- Nhiều seed, key/checkpoint và prompt mới. Tập test hiện đã dùng để định hướng phát triển nên chỉ là development benchmark.
- Phân tích owner sau freeze có thể kiểm tra correction có liên quan residual/watermark không; không dùng lại để chọn kết quả trên cùng test.

## 7. Chạy và chi phí

```bash
# Chỉ SleeperMark với mặc định mới
bash run_blind_quantization.sh --watermark sleepermark

# Cả Stable Signature và SleeperMark
bash run_blind_quantization.sh

# Ablation random continuous context, không PGD
bash run_blind_quantization.sh --watermark sleepermark --methods adversarial_equivariance_qat --probe-steps 0

# Tái lập calibration kiểu cũ cho đối chứng CFG
bash run_blind_quantization.sh --watermark sleepermark --methods cfg_reconstruction --calibration-mode renoised
```

Batch1, FP32, gradient checkpointing, backward preservation trước auxiliary. Không nhân đôi UNet thường trú trên GPU. Nhánh mới tốn thêm teacher forwards và PGD; không claim nhanh bằng nhánh cũ. Chưa đo peak VRAM/thời gian trên H200 hoặc RTX server, không bảo đảm tránh OOM nếu GPU chia sẻ.

Kiểm tra local: 13 test qua, gồm công thức x0 cho epsilon/v-prediction, giới hạn perturbation và correction, gradient ascent trên mô hình giả lập, capture latent trước forward, khôi phục model nguồn và training/export trên UNet nhỏ của diffusers 0.35.1 với gradient checkpointing. Đây là kiểm tra CPU, không thay thế chạy checkpoint thật trên GPU.
