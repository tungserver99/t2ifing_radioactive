# Sau run 000857: residual QAT và đối chứng decoder FP32

## Bằng chứng và điều chưa biết

Run `blind_20260922_000857_531639`: baseline BA 98,54%, TPR 99%; natural residual BA
96,02%, TPR 98%, chỉ có 1/99 ảnh vốn bị phát hiện thoát detector. Đây là tín hiệu giảm
độ chính xác giải mã, chưa phải watermark removal mạnh. Residual tăng lỗi bit từ
70 lên 191/4800; 28% ảnh vi phạm ít nhất một ngưỡng chất lượng theo ảnh.

QAT cũ có 100 update nhưng checkpoint được chọn ở step 0. Step 100 đổi 5,32% code,
SSIM search 0,817 và objective 0,01184, tệ hơn 0,01040 của RTN. Owner evaluation chỉ
đo checkpoint được chọn; không được kết luận checkpoint step 100 giữ watermark.
Không có trace đảo chiều code từng bước, nên chưa chứng minh có oscillation.

## Nguồn và cách áp dụng

[Stable Signature is Unstable, §4.3/5.1](https://arxiv.org/html/2405.07145v1)
fine-tune decoder từ latent ước lượng của ảnh không watermark; dùng pixel/perceptual
loss và discriminator. Paper dùng 4.000 ảnh, optimizer có warm-up và cosine decay.
Pilot 32 ảnh/100 update hiện tại không đủ để xem như đã tái lập kết quả paper.

[Overcoming Oscillations in QAT, ICML 2022](https://proceedings.mlr.press/v162/nagel22a.html)
nghiên cứu dao động giữa các grid point và đề xuất dampening/weight freezing.
Đây là cơ sở để kiểm tra ổn định tối ưu, không phải bằng chứng rằng VAE của chúng ta
đã mắc đúng cơ chế ấy. Code mới dùng LR riêng và lịch giảm LR đơn giản; **chưa triển
khai hay tuyên bố tái lập thuật toán dampening/freezing của paper**.

## Thay đổi thực thi

1. Giữ `natural_residual` làm đối chứng tích cực nhất đã quan sát.
2. Thêm `natural_residual_qat`: cùng frozen residual basis, natural reconstruction,
   LPIPS và bảo toàn phần bù như residual; thay alpha rounding bằng code-offset STE.
   Mỗi forward/evaluation/export đều hard quantized, giới hạn offset ±2 code.
   Trust loss mặc định 0,01; đặt 0 để so cùng objective với residual rounding.
3. QAT có `--qat-lr 0.001` thay vì dùng chung LR alpha 0,01. Warm-up 10 update rồi
   cosine về 10% LR đỉnh. Đây là cấu hình thực nghiệm, không đảm bảo hội tụ hoặc evasion.
4. Thêm `natural_full_finetune`: toàn bộ decoder FP32 gồm bias/norm, MSE + LPIPS;
   LR 1e-5, cùng warm-up/decay, mặc định không bảo toàn RGB ảnh marked.
   `--ft-preserve-weight` có thể thêm low-pass consistency. Đây là adaptation phục vụ
   chẩn đoán, không có discriminator/Watson-VGG và không phải reproduction nguyên paper.

FP32 control được train từ marked decoder độc lập, chỉ chạy một lần kể cả sweep nhiều
bits/clip. Checkpoint mang role `finetune_control`, group `fp32_unrestricted_decoder_control`,
artifact `decoder_fp32.safetensors`; không xuất quantizer giả cho nó. Encoder/UNet giữ nguyên.
Owner mechanism phân tích nó tại FP32 endpoint, tách `parameter_space` với quantized endpoint.

QAT report bổ sung RMS code-offset và tỷ lệ offset có trị tuyệt đối >=1. Khả năng cho
phép đi nhiều ô không có nghĩa run ngắn với LR nhỏ đã đi nhiều ô. Tất cả LR update được log.
Số bước hợp lệ và số bước checkpoint được chọn là hai cột khác nhau; fallback không phải
thành công tối ưu. Test không tham gia lựa chọn; owner metrics chạy sau khi freeze.

## Chạy

```bash
# Pilot: 4 nhánh W4 + 1 control FP32, tự tải model/data và owner-evaluate
bash run_blind_quantization.sh

# Development lớn hơn: natural counts độc lập số prompt
bash run_blind_quantization.sh --natural-train-n 256 --natural-search-n 64 \
  --steps 200 --qat-steps 1000 --ft-steps 1000 --eval-every 50

# Ablation: cùng residual objective, bỏ thêm trust penalty của QAT
bash run_blind_quantization.sh --natural-methods natural_residual natural_residual_qat \
  --qat-trust-weight 0

# Khôi phục loss QAT cũ với optimizer mới
bash run_blind_quantization.sh --natural-methods natural_qat_purification
```

Counts mặc định vẫn 32/20 để so pilot cũ. Mở rộng dữ liệu và budget phải ghi rõ thay đổi;
không quy toàn bộ cải thiện cho method khi thay cùng lúc nhiều yếu tố. Branch compute
được log; benchmark này không mặc nhiên equal-compute. Với nhiều ảnh, chọn cache CPU
hoặc tăng dần; không mặc định nạp toàn bộ ảnh vào GPU.

Report CSV/JSON ở `output_attack`; ảnh/model/checkpoint ở `output_artifacts` như trước.
Không thêm checkpoint theo từng step. Chất lượng không đạt vẫn được ghi và chạy tiếp.

## Quyết định sau run

- FP32 giảm TPR nhiều còn quantization yếu: có bằng chứng về khoảng cách thực nghiệm
  giữa hai không gian tối ưu, nhưng chưa chứng minh grid W4 không tồn tại nghiệm tốt.
- FP32 cũng yếu: kiểm tra data/budget/loss, sau đó cân nhắc discriminator và reproduction
  protocol gốc; không kết luận watermark bền với mọi fine-tune.
- Residual QAT tốt hơn residual rounding: kiểm tra cùng budget/loss, nhiều seeds và
  quality-matched so sánh trước khi claim contribution.

Low-bit artifact hợp lệ không tự động chứng minh thao tác là PTQ thông thường. Code-offset
được tối ưu là action space rộng hơn rounding hai lựa chọn; FP32 fine-tune rộng hơn nữa.
Không gọi các hướng đó cùng một threat model deployment-strict.

Tập test đã được dùng để phát triển method nên coi là development. Sau khi chốt protocol,
dùng prompt/seed mới và nhiều run; không điều chỉnh FPR/ngưỡng để làm đẹp attack.
