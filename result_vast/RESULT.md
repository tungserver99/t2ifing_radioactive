# Tổng hợp kết quả semantic fingerprint: SD2.1 → SD1.4

## 1. Kết quả cũ — threshold_cfg_v1

Run: `run_semantic_fp_20261020_135117_009783`. Các bảng trong mục này thuộc phương pháp continuous-score/threshold CFG cũ, không phải Semantic Sign Fingerprint.

Teacher/student use their respective frozen clean 1% FPR null thresholds.

| Model | Detection | Mean score | Bit accuracy | FID | KID | CLIP |
|---|---:|---:|---:|---:|---:|---:|
| teacher_clean_test | 1.00% | -0.01394 | 37.5% | -2.911e-05 | -0.00077483 | 0.78625 |
| teacher_wm_test | 100.00% | 1.9916 | 100.0% | 79.634 | 0.024161 | 0.77733 |
| unadapted | 0.90% | -0.06103 | 37.5% | 50.286 | -0.00016148 | 0.78217 |
| lora_r32_conv_wm | 99.70% | 1.5117 | 100.0% | 62.676 | 0.0069348 | 0.79334 |
| full_wm | 99.90% | 1.6457 | 100.0% | 75.077 | 0.016586 | 0.78809 |

## Paired teacher validation quality

| Kappa | Selected | Clean CLIP | Fingerprinted CLIP | CLIP drop | Paired LPIPS | Quality pass |
|---|---|---:|---:|---:|---:|---|
| 1 | False | 0.78202 | 0.78587 | -0.0038546 | 0.488 | True |
| 1.5 | False | 0.78202 | 0.77818 | 0.003841 | 0.52049 | True |
| 2 | True | 0.78202 | 0.77279 | 0.0092314 | 0.54189 | True |

Teacher paired quality details: teacher_quality.json / teacher_quality.csv. Student checkpoint curves: checkpoint_metrics.csv.

- One training seed; image confidence intervals do not measure training-seed uncertainty.
- Unadapted baseline contrasts do not isolate generic fine-tuning effects without clean-trained controls.
- Teacher strength uses held-out teacher validation only; students selected solely by denoising MSE.

## 2. Kết quả mới — sau khi chuyển sang Semantic Sign Fingerprint / RSSF (`semantic_sign_v1`)

Run: `run_semantic_fp_20261021_174725_058270`.

Bản sửa theo `semantic_sign_fingerprint_codex_spec.md` chuyển từ continuous-score/threshold CFG sang fingerprint dấu của 32 semantic coordinates, được center bằng train medians. Verifier chính đếm số bit khớp secret và dùng exact binomial tail: với 32 bit, alpha = 0.01, cần ít nhất 24 bit khớp để phát hiện một ảnh. Chỉ số này là **analytic detection**, không phải empirical TPR@1%FPR được calibrate trên clean images.

Teacher vẫn train bằng LoRA rồi merge để sinh ảnh bằng pipeline thông thường. Loss mới gồm preservation, squared hinge trên các bit được chọn bằng clean coordinates, cùng LPIPS và pixel MSE trên ảnh dự đoán x0 tại các timestep cuối. Ba candidate được train độc lập với coordinate margin μ = 0.1, 0.2, 0.3. μ của bản mới không tương đương κ của bản cũ.

### Teacher validation

Số liệu được chép từ output đọc `teacher_fp/candidates.json` trên server mà người dùng đã gửi trong cuộc trò chuyện.

| Coordinate margin μ | Analytic detection rate | CLIP drop | Paired LPIPS | Detection ≥ 90% | CLIP drop ≤ 0.01 | LPIPS ≤ 0.15 | Đạt toàn bộ gate |
|---|---:|---:|---:|---|---|---|---|
| 0.1 | 99.8% | 0.013711 | 0.407089 | Đạt | Không đạt | Không đạt | Không |
| 0.2 | 100.0% | 0.012716 | 0.415702 | Đạt | Không đạt | Không đạt | Không |
| 0.3 | 100.0% | 0.017038 | 0.438035 | Đạt | Không đạt | Không đạt | Không |

Giá trị gốc chưa làm tròn:

```text
mu=0.1 TPR=0.998 CLIP_drop=0.013711029722690586 LPIPS=0.4070892530158162
mu=0.2 TPR=1.0   CLIP_drop=0.01271555155515669  LPIPS=0.4157024439424276
mu=0.3 TPR=1.0   CLIP_drop=0.017037655413150832 LPIPS=0.4380349085479974
```

`TPR` là tên hiển thị của lệnh đọc `metrics.detection_rate`; báo cáo dùng tên analytic detection rate để tránh nhầm với TPR tại empirical clean FPR = 1%.

### Kết luận và trạng thái chạy

- Watermark detection trên teacher validation đạt 99.8–100%, nhưng cả ba candidate đều trượt **cả CLIP drop lẫn LPIPS**. Không candidate nào đủ điều kiện được chọn.
- Chương trình dừng tại `01e_train_native_teacher` với thông báo `No RSSF teacher passed analytic TPR/CLIP/LPIPS gates; students stopped`. Đây là dừng theo quality gate, không phải lỗi OOM trong traceback đã gửi.
- Run mới này chưa đi tiếp sang train/eval student. Kết quả student trong mục 1 thuộc run cũ, không được dùng làm kết quả RSSF.
- Giảm μ từ 0.3 xuống 0.1 giúp LPIPS giảm từ khoảng 0.438 xuống 0.407, nhưng vẫn vượt xa gate 0.15. Chưa có bằng chứng rằng chỉ giảm margin sẽ giải quyết được vấn đề chất lượng.
- LPIPS mới thấp hơn về mặt số học so với teacher validation cũ (0.488–0.54189). Chưa đối chiếu đầy đủ provenance của hai run nên đây chưa phải kết luận so sánh có kiểm soát.

### Nhận định kỹ thuật cần kiểm chứng

Kiểm tra code cục bộ cho thấy LPIPS dùng ảnh RGB trong [-1,1], validation kiểm tra cùng prompt/seed, và loss LPIPS có đường gradient về teacher. Chưa xác định được lỗi tính metric từ thông tin hiện có.

Giả thuyết cần kiểm chứng: loss chất lượng lúc train so sánh ảnh dự đoán x0 từ cùng một latent nhiễu tại một timestep cuối, còn validation so sánh ảnh sinh qua toàn bộ 50 bước từ Gaussian noise. Giữ gần nhau ở phép dự đoán một bước không bảo đảm hai quỹ đạo sinh ảnh hoàn chỉnh vẫn gần nhau. Đây chưa phải nguyên nhân đã được xác nhận.

Cần đối chiếu `lpips`, `pixel_mse`, `fp`, `preserve` trong training logs với native validation LPIPS để phân biệt khoảng cách train/inference với cân bằng loss chưa phù hợp. Chưa nên tự nới quality gate hoặc chạy lại nguyên cấu hình.

### Nguồn dữ liệu và giới hạn lưu trữ

Người dùng không còn truy cập server. Mục kết quả mới được tổng hợp từ số liệu và traceback đã gửi trong cuộc trò chuyện; không giả định artifact của run mới đã được tải về máy cục bộ. Chưa có thêm dữ liệu để ghi FID, KID, absolute CLIP, bit recovery, clean empirical FPR hoặc training loss curves của run mới. Student metrics chưa được tạo vì pipeline đã dừng trước bước train student.

Đường dẫn server trước đây để tra cứu nếu khôi phục được dữ liệu:

```text
/workspace/t2ifing_radioactive/run_semantic_fp_20261021_174725_058270/
  teacher_fp/candidates.json
  logs/01e_train_native_teacher.log
  images/teacher_clean_validation/
  images/candidate_mu_0.1_validation/
  images/candidate_mu_0.2_validation/
  images/candidate_mu_0.3_validation/
```
