# SleeperMark trên server: kiểm tra run 20260927_061035_583957

## Trạng thái quan sát

Run dùng commit `ed6919d`, H200, chạy trong tmux `thinhnd_sleepermark`.
Ba nhánh học đã hoàn thành 2.000 bước. Tại lúc kiểm tra đầu tiên, owner evaluation
chưa xong; không có lỗi OOM. Baseline đã hoàn tất cả 100 ảnh triggered và 100 ảnh
ordinary: BA **99.5833%**, TPR **100/100**, ordinary detection **0/100**.
Detector double-tail 48 bit, inclusive ≥36 hoặc ≤12, FPR tổng mục tiêu 0.001.

Kết quả attack đầu tiên đã hoàn tất trong lúc review: `fixed_ptq_w4` đạt BA
**99.0417%**, TPR **100/100**, ordinary detection **0/100**. Chất lượng ordinary
PSNR/SSIM **20.338/0.704**, triggered **19.895/0.688**; chỉ 9/100 ảnh triggered
qua đồng thời ngưỡng PSNR≥25 và SSIM≥0.8; joint success 0/100. So với fixed W4
run trước (BA 99.333%, ordinary SSIM 0.682, triggered SSIM 0.679), MSE init tăng
chất lượng một ít và giảm BA một ít, nhưng hoàn toàn chưa tạo evasion.

| Nhánh | Thời gian TRAIN | Peak allocated VRAM | Chẩn đoán cuối TRAIN |
|---|---:|---:|---|
| CFG reconstruction W4 | 15.53 phút | 7.33 GiB | Mean-layer code-change 2.205% |
| Equivariance W4 | 32.65 phút | 7.86 GiB | Mean-layer code-change 2.236% |
| FP32 base + delta4 | 31.51 phút | 8.16 GiB | Nonzero delta codes 5.789% |

Hai loại code fraction có mẫu số khác nhau: trung bình theo layer đối với W4,
tổng số phần tử đối với delta. Không trực tiếp so sánh các phần trăm này.
Spatial correction RMS ở mẫu log cuối là 0.001852, thấp hơn cap 0.05 nhiều lần.
Đây là một minibatch, không phải thống kê toàn tập. Tăng cap chưa chắc thay đổi
target. Loss ở các bước log dùng minibatch/timestep khác nhau, không thể dùng
chênh lệch loss đầu/cuối để khẳng định hội tụ hoặc suy giảm watermark.

Các JSON/CSV hiện có đã được tải về `output_attack/sleepermark_20260927_061035_583957`.
Đây là snapshot tạm thời, không tự đồng bộ và không chứa checkpoint/ảnh.

Kết quả hoàn chỉnh trước đó `20260927_013638_836714` vẫn là đối chiếu: tất cả nhánh
joint success 0/100; equivariance TPR 99%, còn các nhánh khác 100%. Delta giữ ảnh
tốt hơn (triggered SSIM 0.897) nhưng không phá watermark tốt hơn. Chưa được thay
các số đó bằng suy đoán từ loss của run đang chạy.

## Cơ sở nghiên cứu và khoảng trống

[SleeperMark, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_SleeperMark_Towards_Robust_Watermark_against_Fine-Tuning_Text-to-image_Diffusion_Models_CVPR_2025_paper.pdf)
gắn watermark với trigger văn bản và tách hành vi này khỏi sinh ảnh thông thường.
Vì vậy calibration chỉ trên prompt thường có thể không kích hoạt phần cần xóa.
Random punctuation hoặc spatial defect lớn không chứng minh đã tìm thấy trigger.

[AccuQuant, NeurIPS 2025](https://arxiv.org/abs/2510.20348) chỉ ra lợi ích của học
qua nhiều bước denoising để xử lý quantization error tích lũy. Đây là tiền lệ cho
multi-step calibration, không phải bằng chứng xóa watermark hoặc novelty của ta.
Implementation dưới đây là biến thể thử nghiệm với target và truncated gradient
riêng, không phải reproduction đầy đủ của AccuQuant.

## Nhánh mới: conditional_rollout_qat

Giữ các nhánh trước làm đối chứng; thêm một nhánh W4 group 64, chỉ học quantizer.
Không fine-tune trọng số gốc, không dùng clean model/key/extractor/trigger lúc học.

Gọi `u` và `c` là predicted x0 của marked teacher với empty/ordinary hoặc augmented
TRAIN context. Đặt `r = c-u`, và `A(r)` là trung bình dự đoán ở hai latent dịch
ngược chiều, căn chỉnh về hệ tọa độ ban đầu. Target là:

```text
x0_target = u + guidance*r + cap_RMS(mask * guidance * highpass(A(r)-r))
```

Khác target spatial cũ, phép hiệu chỉnh không tác động trực tiếp lên spatial defect
của `u`. Sai lệch không gian độc lập conditioning triệt tiêu trong `c-u`;
conditional texture vẫn có thể bị ảnh hưởng. Không gọi đây là ownership subspace.
Border được giữ nguyên, correction có cap như nhánh cũ.

Từ cùng một latent TRAIN, chạy hai đường DDIM eta=0 riêng: teacher dùng target đã
hiệu chỉnh, student dùng UNet W4 hiện tại. Tại từng bước, so sánh predicted x0 của
hai đường bằng loss chuẩn hóa theo loại prediction; backprop xong mới detach và
chuyển sang timestep tiếp. Như vậy bước thứ hai quan sát cả sai lệch state tích lũy.
Đồng thời giữ ordinary CFG reconstruction loss cũ trên toàn miền timestep.

- Mặc định horizon=2; gần cuối trajectory có thể chỉ còn một bước hợp lệ.
- Gradient **bị cắt giữa các state**, không backprop toàn trajectory; bộ nhớ activation
  không tăng tuyến tính theo horizon. Mỗi bước thêm các teacher/student forward.
- Bắt đầu từ latent ordinary TRAIN: hai bước với augmented context không tương
  đương sinh toàn trajectory với context ấy. Vẫn có hạn chế off-distribution.
- Dùng scheduler DDIM inference thật, giữ clipping/alpha endpoint/steps offset,
  hiện chỉ hỗ trợ timestep spacing `leading`.
- Inference không thêm dịch ảnh, JPEG, resize hay sửa prompt; artifact vẫn là trọng
  số UNet được quantize. Nhánh delta cũ vẫn được ghi riêng là FP32 base + delta4.
- Checkpoint là bước cuối khai báo trước; owner chỉ đánh giá sau khi freeze.

Đây là giả thuyết cải tiến có thể kiểm chứng, **chưa phải kết quả đột phá đã đo**.
Nếu nhánh mới chỉ tăng SSIM mà TPR vẫn 100%, kết luận là cải thiện utility, không
phải xóa watermark. Tránh tiếp tục thêm ngân sách khi proxy không liên quan attack.

## Lệnh chạy và kiểm chứng

Mặc định SleeperMark thêm nhánh mới, giữ bốn nhánh cũ. Chạy riêng nhánh mới:

```bash
bash run_blind_quantization.sh --watermark sleepermark --methods conditional_rollout_qat
```

Đối chứng cơ chế: cùng seed, code LR, group size, steps, TRAIN/TEST, so
`equivariance_qat` với `conditional_rollout_qat --rollout-horizon 1` (đổi target),
rồi horizon 1 với 2 (đổi rollout). Cùng steps không đồng nghĩa cùng FLOPs; báo cả
thời gian và `logical_unet_forwards` và cần thêm đối chứng cùng ngân sách compute.

Đo BA, TPR double-tail, quality trên ảnh triggered/ordinary, joint success và FID.
FID 100 ảnh với marked reference chỉ là chẩn đoán, không thay FID real lớn.
Các run đã xem để phát triển phương pháp là development; xác nhận cần prompt/seed
mới và tốt hơn nữa là key/checkpoint mới, không chọn lại checkpoint từ owner TEST.

Kiểm tra ban đầu: 24 test CPU pass, gồm UNet Diffusers thật, gradient checkpointing,
export/restore trọng số, epsilon/v/sample roundtrip, triệt tiêu spatial bias độc
lập conditioning, hai state teacher/student thực sự khác nhau ở bước hai, không
lặp lại terminal timestep. Bash syntax pass. CUDA smoke test trên H200 với UNet
Diffusers nhỏ cũng pass: batch 2, bốn optimizer steps, hai DDIM transitions,
24 ma trận thay đổi, trọng số gốc khôi phục đúng hash. Peak allocated 0.067 GiB
là của model test nhỏ, **không phải** ước tính VRAM cho SD1.4 đầy đủ. Các test
không chứng minh attack mạnh.

Code cải tiến được kiểm tra trong worktree riêng trên server; job full cũ tiếp
tục chạy đúng commit ban đầu. Chưa khởi chạy một sweep full thứ hai song song.

Sau đó `conditional_rollout_qat` được khởi chạy song song trong tmux riêng. Hai
process quan sát lúc calibration dùng tổng khoảng 11.6 GiB, GPU đạt 100%, còn
trống khoảng 131.6 GiB. Điều này xác nhận không bị giới hạn VRAM, đồng thời cảnh
báo rằng thêm process sẽ tranh compute; mục tiêu của parallel suite là giảm wall
time toàn suite, không bảo đảm giảm thời gian từng branch.

Launcher mới chạy các branch SleeperMark thành process độc lập. Số process được
tính từ VRAM trống, phần dự phòng và ước lượng per-process; không kiểm tra tên GPU.
Mỗi branch có log/result riêng, rồi tạo `parallel_summary.csv/json`. Calibration
tensor được cache lên CUDA nếu phép kiểm tra dung lượng an toàn pass, bảo toàn
shared tensor references. Batch size và optimizer budget không tự thay đổi.
