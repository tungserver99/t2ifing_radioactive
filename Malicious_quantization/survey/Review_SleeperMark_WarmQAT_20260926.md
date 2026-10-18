# Review và thử nghiệm tiếp theo (26/09/2026)

## Kết quả SleeperMark

Run `sleepermark_20260926_052518_273157`, 100 ảnh test, detector double-tail FPR tổng 0.001:

| Nhánh | Bit accuracy | TPR | SSIM ảnh thường | FID ảnh thường so với marked |
|---|---:|---:|---:|---:|
| Marked baseline | 99.58% | 100% | 1 | — |
| Fixed W4 | 99.06% | 100% | .634 | 114.87 |
| Model reconstruction W4 | 97.60% | 100% | .616 | 113.63 |
| Natural rounding W4 | 98.21% | 100% | .576 | 125.93 |
| Natural finetune W4 | 98.71% | 100% | .598 | 120.51 |
| Natural joint finetune W4 | 98.65% | 100% | .621 | 104.76 |

Attack thất bại ở tất cả nhánh đã đo; giảm chất lượng đáng kể. FID chỉ dùng 100 ảnh và reference là marked model, không phải FID so với phân phối ảnh thật. SSIM ở bảng là ảnh thường, còn watermark đo trên ảnh có trigger: không ghép hai metric thành joint success trên cùng ảnh.

## Stable Signature: centered warm-QAT

Giữ implementation `natural_joint_quality_finetune` nhưng bỏ khỏi profile transfer. Thêm lại `natural_residual_qat_warm` với chế độ `centered_scale`.

Khởi tạo từ mã residual đã được chọn trên SEARCH. Đặt `q0` là các mã đó, tối ưu `Q = s0 exp(a) round(clamp(q0 + d, -8, 7))`, dùng STE, `d` giới hạn ±2, scale trong [0.8,1.25] lần scale ban đầu. Điểm bắt đầu khớp chính xác residual. Trust penalty lấy mốc residual thay vì kéo về trọng số marked gốc. LR mã .01, LR scale .001, có warmup/cosine. Thêm penalty PSNR/SSIM từng ảnh trên tập preservation và SEARCH, không dùng detector.

Đây là mở rộng không gian quantizer, không phải huấn luyện toàn bộ decoder. So sánh với legacy phải ghi rõ scale được học và tâm giới hạn offset đã thay đổi. Log `code_change_fraction_vs_warm`, `scale_log_rms`, `warm_quality_penalty` giúp kiểm tra có thật sự di chuyển trên lưới hay không. Chế độ legacy vẫn có qua `--warm-qat-mode legacy`.

## SleeperMark: sửa calibration và thử prefix consistency

Mặc định mới chạy ba nhánh trên cùng W4 group size 64:

1. `fixed_ptq`: RTN làm đối chứng cùng độ hạt lượng tử hóa.
2. `cfg_reconstruction`: giữ dự đoán CFG=7.5 và nhánh unconditional của marked UNet trên prompt TRAIN. Đây là đối chứng chất lượng.
3. `prefix_consistency_qat`: cùng loss trên, thêm yêu cầu dự đoán với prefix dấu câu ngẫu nhiên gần dự đoán prompt thường của marked model. Prefix sinh độc lập từ bảng dấu câu, không đọc trigger/key/extractor. Prefix loss bắt đầu sau 20% số bước, tăng dần đến trọng số .25 ở 50%.

Giả thuyết: hạn chế độ nhạy với prefix không mang nội dung có thể làm yếu cơ chế kích hoạt mà giữ ngữ nghĩa. **Chưa biết có tổng quát sang trigger của SleeperMark không.** Có thể chỉ làm giảm năng lực hiểu dấu câu; cần đo prompt thông thường và prompt có dấu câu ngoài TRAIN. Không gọi đây là breakthrough hay xóa watermark trước khi owner evaluation xác nhận.

Calibration vẫn dùng latent ảnh cuối được thêm noise theo timestep DDPM ngẫu nhiên; chưa phải cache toàn trajectory inference. CFG reconstruction khắc phục một sai lệch, chưa giải quyết toàn bộ distribution shift. Các backward chạy nối tiếp và dùng activation checkpointing, batch mặc định 1; chưa đo peak VRAM thực trên server.

Các nhánh cũ vẫn chạy qua `--methods`; `--quant-group-size 0` dùng quantizer per-channel cũ. `--prefix-weight 0` là ablation bỏ prefix objective. Không so tốc độ như bằng nhau: prefix có thêm một forward/backward mỗi bước. Grouped W4 có nhiều scale hơn per-channel, vẫn là fake quantization FP32, chưa phải kernel INT4.

## Chạy

```bash
bash run_blind_quantization.sh
```

Chạy riêng SleeperMark:

```bash
bash run_blind_quantization.sh --watermark sleepermark
```

Checkpoint được đóng băng trước owner evaluation, không chọn theo kết quả test. Chỉ khi TPR giảm ở chất lượng tương đương và lặp lại trên nhiều seed mới có cơ sở kết luận cải tiến.

## Cơ sở nghiên cứu

- [LSQ](https://arxiv.org/abs/1902.08153): học step size đã có tiền lệ; không nhận đây là novelty riêng.
- [Q-Diffusion](https://arxiv.org/abs/2302.04304): calibration diffusion cần chú ý phân phối qua timestep; không chứng minh hiệu quả xóa watermark.
- [SleeperMark, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_SleeperMark_Towards_Robust_Watermark_against_Fine-Tuning_Text-to-image_Diffusion_Models_CVPR_2025_paper.pdf): watermark thiết kế để chống fine-tuning; kết quả thất bại ở đây phù hợp với việc cần kiểm tra cơ chế kích hoạt thay vì chỉ tăng bước natural denoising.
