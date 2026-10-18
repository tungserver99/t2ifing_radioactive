# Revision 2026-09-22: quantization theo residual tái tạo

## Bằng chứng từ pilot

Nguồn local: `output_attack/owner_evaluation.json`, `report.json`, `search.csv` và
`watermark_retention.csv`, thuộc run `blind_20260921_154529_369243`.
Các file tổng được copy ở root output_attack; chưa có manifest ở root để tự kiểm
chứng toàn bộ frozen run. Phân tích dưới đây dựa trên số liệu báo cáo, không phải
một lần chạy lại extractor trên ảnh.

- Reference BA 98.50%, TPR 99/100; không candidate nào làm mất detection trên 99
  ảnh vốn được phát hiện. Tổng evasion 1% là lỗi baseline đã tồn tại.
- Natural rounding W4: BA 97.25%, TPR 99%, PSNR 32.94, LPIPS 0.02436.
- Rounding-scale W4: BA 97.3125%, TPR 99%; highpass energy ratio 0.5058.
- Pseudo-target blur: BA 98.1458%, TPR 99%; target vẫn giữ watermark.
- Random W8/W4 đều chọn step 0 RTN fallback. Sensitivity W8/W4 lần lượt mất
  1593.10 và 1588.48 giây, tổng 53.03 phút / 108.63 phút attack (~48.81%).

Vì thế không suy ra tăng step sẽ giải quyết được objective sai hướng. Giảm BA
trung bình không đủ: phải xem detection từng ảnh với ngưỡng 36/12 bit.
Không quy đổi chênh lệch BA thành một hệ số cần tăng attack: detection phi tuyến
và phụ thuộc phân phối từng ảnh. Pilot này là dữ liệu phát triển phương pháp.

## Cơ sở nghiên cứu và giới hạn suy luận

1. [Hu et al., Stable Signature is Unstable](https://arxiv.org/html/2405.07145v1),
   §4.2–4.3: dùng ảnh natural, latent ước lượng từ encoder, fine-tune decoder với
   reconstruction/perceptual và adversarial training. Không cần clean counterpart
   của output sinh. Nghiên cứu đó cho phép cập nhật decoder tự do; implementation
   ở đây chỉ học rounding, không tuyên bố tái lập thuật toán hoặc kết quả paper.
2. [AdaRound](https://arxiv.org/abs/2004.10568): task-aware rounding và soft relaxation.
   Code hiện tại dùng hard-forward sigmoid STE, không phải triển khai AdaRound gốc.
3. [BRECQ](https://arxiv.org/abs/2102.05426): reconstruction theo block là baseline
   PTQ hữu ích. Mục tiêu của nó giữ hành vi mạng, không tự tạo loss xóa watermark.
4. [When There Is No Decoder](https://arxiv.org/html/2507.03646v1), §3.5:
   nghiên cứu surrogate extractor/fine-tuning trong no-box. Hướng đó thêm surrogate
   đã huấn luyện và không chứng minh residual PCA là ownership subspace.

Residual từ D_w(E_w(x))-x chứa cả watermark, lỗi autoencoder và texture. Thành phần
phổ biến nhất hoặc phương sai lớn nhất KHÔNG mặc nhiên là watermark. Đề xuất dưới
đây là giả thuyết thực nghiệm có ablation, chưa có bằng chứng cải thiện TPR hoặc
novelty đủ cho một paper. Không sử dụng detector/key trong fit/optimization/search.

## Nhánh mới: natural_residual

Chỉ dùng natural TRAIN để lấy residual của VAE marked FP32. Chia residual thành
patch RGB 8x8 không chồng lấn; trừ DC từng kênh, lấy tối đa 256 patch/ảnh với RNG
cố định. Tích lũy second moment uncentered bằng CPU FP64 rồi lấy tối đa 8 eigenvector
có eigenvalue dương đủ lớn. Uncentered giữ tín hiệu mean lặp lại; DC removal tránh
chiếm rank chỉ bởi lệch màu. Orthonormal basis B đóng băng trước search/test.
Không resize/JPEG hậu xử lý ảnh attack; patching chỉ phục vụ loss.

Với p(error) là vector patch dài d=192 và rank r:

    L_proj(error) = mean_patches(||B p(error)||² / r)
    delta = D_Q(z_generated) - D_w(z_generated)
    L_pres(delta) = mean(||(I - BᵀB)p(delta)||² / d) + 0.1 * MSE(delta)
    L = MSE(D_Q(E_w(x)), x) + 0.1*LPIPS(D_Q(E_w(x)), x)
        + residual_weight*L_proj(D_Q(E_w(x)) - x) + 2*L_pres(delta)

Phép chia r, thay vì d, tăng trọng số các hướng đã chọn tương đối với pixel MSE;
đây không phải whitening hoặc loss được chuẩn hóa bằng noise watermark.
Loss bảo toàn vẫn phạt đầy đủ phần bù và phạt nhẹ cả RGB, giảm xung đột do bắt
output mới tái tạo nguyên cả residual marked. Vẫn có nguy cơ làm hỏng texture.
Encoder/decoder source/bias/norm/UNet không học; chỉ alpha rounding học. Scale cố
định để so trực tiếp với natural_rounding. Export hard grid và inference như cũ.

Search natural riêng dùng MSE + 0.1 LPIPS + residual_weight L_proj, với cùng basis
TRAIN. Loss preservation dùng generated TRAIN, quality gate dùng generated SEARCH.
Training dùng raw RGB reconstruction; search clamp RGB về [0,1] như nhánh natural
cũ. Chính sách report tiếp tục ghi tất cả chất lượng thấp, không tự đổi threshold.

`residual_calibration.json` ghi effective rank, captured energy, mean energy,
overlap giữa hai basis fit từ hai nửa TRAIN (theo chẵn/lẻ). Overlap thấp là dấu
hiệu proxy thiếu ổn định; overlap cao vẫn không xác nhận nó là watermark.
`residual_basis.safetensors` lưu ngoài báo cáo trong `output_artifacts/checkpoints`, được hash
cùng artifact trước test. Fit streaming từng ảnh không giữ graph/activation cả tập.

## Protocol và ablation

Mặc định còn 5 nhánh W4: fixed_ptq, rounding_scale (model-only cũ), reconstruction,
natural_rounding (đối chứng), natural_residual (giả thuyết mới). Sensitivity, rounding,
W8 và natural_rounding_scale vẫn gọi được tường minh, không mặc định chạy.
Không tăng steps hoặc bit severity mặc định, để tránh đồng thời thay quá nhiều yếu tố.

```bash
bash run_blind_quantization.sh

# Ablation A: projected residual loss, full preservation như natural_rounding
bash run_blind_quantization.sh --residual-preservation full

# Ablation B: đổi preservation nhưng không projected loss
bash run_blind_quantization.sh --residual-weight 0

# Sanity: residual branch phải tương đương natural_rounding trong cấu hình này
bash run_blind_quantization.sh --residual-weight 0 --residual-preservation full

# Stress-test bitwidth; tách comparison_group, không chọn bằng owner metrics
bash run_blind_quantization.sh --bits 4 3
```

Mỗi lệnh tạo run mới, tự evaluate, giữ báo cáo/ảnh/checkpoint tách biệt. Run 100 ảnh
là pilot. Sau khi khóa thiết kế cần prompt test mới, seed mới và nhiều model/key;
đổi seed đơn thuần không làm prompt cũ thành bộ test độc lập hoàn toàn.
Nhánh natural gồm thêm dữ liệu ngoài và LPIPS; không được gọi là model-only.
Nếu residual branch chỉ giảm projected loss mà TPR vẫn 99%, phải kết luận proxy
không transfer; không tiếp tục chọn rank/lambda trên cùng owner test rồi gọi blind.
Kiểm chứng phương pháp cần đánh giá double-tail evasion, paired bit errors, chất
lượng và runtime; không dùng natural search objective để thay cho owner outcome.
