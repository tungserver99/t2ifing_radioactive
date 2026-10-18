### **watermark uu tien**

| Watermark | Loại | Repo chính thức | Backbone | Vì sao ưu tiên |
| ----- | ----- | ----- | ----- | ----- |
| **Stable Signature** | **Decoder-embedded** | **`facebookresearch/stable_signature`** | **SD (VAE decoder, tách rời UNet)** | **Code Meta, đã có sẵn số liệu quantization benign (0.99 bit-acc @ 4-bit) để so sánh trực tiếp; kiến trúc đơn giản để hiểu và lam pipeline PTQ** |
| **AquaLoRA** | **LoRA-embedded (white-box)** | **`Georgefwt/AquaLoRA`** | **SD 1.5 (UNet qua LoRA)** | **Code chính thức ICML 2024; tự nhận diện là "white-box protection" nên rất khớp với threat model quantization (attacker có full access); dễ tạo K surrogate instance**&nbsp; |
| **Gaussian Shading** | **Latent/initial-noise** | **`bsmhmmlf/Gaussian-Shading`** | **SD 1.4/2.0/2.1** | **Code chính thức CVPR 2024; watermark không nằm trong trọng số model — quantization trọng số sẽ tác động gián tiếp hơn (qua lỗi lan truyền vào latent)** |
| **Tree-Ring** | **Latent/initial-noise** | **`YuxinWenRick/tree-ring-watermark`** | **SD (arbitrary, đã test cả 256×256 DDPM)** | **Code chính thức NeurIPS 2023, dùng rộng rãi nhất làm baseline trong toàn ngành; không nằm trong trọng số như Gaussian Shading** |
| **Cert-LAS** | **Certified (smoothing-based)** | **Tác giả công bố "code available" trong paper (ICML 2026\)** | **UNet (qua diffusion classifier)** | **Có code nhưng vừa công bố (2026), độ ổn định/tài liệu hoá chưa rõ; đây là target giá trị cao nhất về mặt narrative (có certificate)** |

### 

### **Bảng 2 — Diffusion backbone nên ưu tiên**

| Ưu tiên | Model | Vì sao |
| ----- | ----- | ----- |
| **1** | **Stable Diffusion 1.4/1.5** | Backbone chung của **gần như mọi** watermark method ở Bảng 1 (Stable Signature, AquaLoRA, Gaussian Shading, Tree-Ring, WatermarkDM) → cho phép **so sánh chéo trên cùng một backbone**, không cần train lại nhiều lần; cũng là backbone mà Q-Diffusion/MixDQ dùng để báo cáo PTQ benchmark, nên có sẵn cấu hình calibration/PTQ tham chiếu |
| **2** | **SD 2.0/2.1** | Gaussian Shading hỗ trợ sẵn; dùng để test tính tổng quát hoá của attack qua kiến trúc UNet hơi khác (attention khác SD1.x) |
| **3 (stretch)** | **PixArt-α (DiT-based)** | FingerInv có test trên đây; Q-DiT cũng dùng DiT — nếu muốn claim tổng quát hoá sang kiến trúc Transformer-based diffusion (khác U-Net), đây là lựa chọn duy nhất có cả watermark-target lẫn PTQ-baseline sẵn |

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

&nbsp;

Cac dang watermark

| Loại watermark | Cơ chế | Đại diện | Chống quantization benign | Chống quantization malicious/adversarial |
| ----- | ----- | ----- | ----- | ----- |
| **Backdoor-based / trigger watermark** | Fine-tune model để phản ứng đặc biệt với trigger prompt/token bí mật | WatermarkDM, SleeperMark | 🟡 Chưa có số liệu quantization công bố trực tiếp (SleeperMark chỉ test robustness với fine-tuning, không test quantization) | ❓ Chưa test |
| **Decoder-embedded watermark** | Fine-tune riêng decoder (VAE) của LDM để mọi ảnh sinh ra mang bit-string | Stable Signature | 🟢 Rất tốt — 0.99 bit-acc ở cả 8-bit lẫn **4-bit** (naive min-max rounding) | ❓ Chưa test |
| **LoRA-embedded watermark** | Gắn module LoRA watermark riêng vào UNet, tách khỏi trọng số gốc | AquaLoRA, AuthenLoRA | 🟢 Tốt — AuthenLoRA báo cáo robust qua Fig.6; cấu trúc LoRA tách biệt giúp watermark ít bị "pha loãng" khi quantize phần backbone | ❓ Chưa test |
| **Latent/initial-noise watermark** | Nhúng pattern vào không gian tần số của noise khởi tạo; verify bằng DDIM/EDICT inversion | Tree-Ring, Gaussian Shading, GaussMarker | 🟡 Không sửa trọng số model → ít bị ảnh hưởng trực tiếp bởi quantization *trọng số*, nhưng **rất dễ vỡ với attack ở tầng ảnh** (theo WAVES: Tree-Ring là watermark dễ bị adversarial image-attack nhất trong benchmark) | ❓ Chưa test ở tầng quantization (khác loại threat) |
| **Adversarial-hiding watermark** | Nhúng watermark mạnh ở bước trung gian, dùng optimization để giấu khỏi ảnh cuối | ROBIN | 🟡 Chưa có số liệu quantization công bố | ❓ Chưa test |
| **Intrinsic fingerprint (non-invasive)** | Không sửa tham số model — dùng inversion/latent code đặc trưng của chính model để nhận diện | FingerInv, TrajPrint, DiffIP | 🟢 Cao trên giấy (FingerInv: 100% ở FP16/BF16; TrajPrint: \>0.95 bit-acc) — **nhưng đây là điểm yếu nhất trong bảng này**: cả hai chỉ test format-cast FP16/BF16, gần như lossless, **chưa hề test INT4/INT8 PTQ thật** dù chính threat model của họ liệt kê "quantization" là attacker capability | ❌ Chưa test — và nền tảng test hiện tại (chỉ FP16/BF16) là yếu nhất trong toàn bảng |
| **Certified watermark (smoothing-based)** | Watermark \+ randomized smoothing theo layer, có chứng minh toán học về bán kính chịu nhiễu | Cert-LAS | 🟢 Tốt — VSR=1.000 ở W8A32/W8A8 (torchao), chỉ sụp ở W4A32 lúc chất lượng ảnh đã hỏng | 🟡 **Có certificate**, nhưng certificate chứng minh cho nhiễu Gaussian/Mahalanobis-ball, **chưa chứng minh cho tập sai số quantization-realizable cụ thể** — có thể phủ hoặc không, chưa kiểm chứng |
| *(Tham chiếu ngoài diffusion)* Watermark DNN classifier tổng quát | Đủ loại (weight-based, trigger-based) | Uchida, T2S, LineageMark | 🟢 Rất tốt — T2S: 100% WSR ở 4-bit; LineageMark: \>90% WER ở 4-bit | ❌ Đây chính là nơi malicious quantization (QCB, QuRA) đã **chứng minh phá được** loại model này (dù target là backdoor, không phải watermark IP trực tiếp) |

### **Nhận xét tổng hợp theo hàng dọc "malicious/adversarial"**

Cột cuối gần như toàn dấu ❓ — đây là bằng chứng rõ nhất cho kết luận ở file survey: **không một loại watermark diffusion nào trong 7 loại trên từng được kiểm chứng chống lại một quantizer bị điều khiển có chủ đích.** Duy nhất hàng \#8 (ngoài phạm vi diffusion) cho thấy dấu ❌ thật — vì đó chính là nơi QCB/QuRA đã chứng minh tấn công thành công trên kiến trúc tương tự (dù mục tiêu là cấy backdoor, không phải xoá watermark).

Nếu muốn, tôi có thể cập nhật bảng này thẳng vào file survey (thêm một mục "Phần III-bis: Bảng theo loại watermark" ngay sau Phần III hiện có) để file tự đứng độc lập không cần đọc lại đoạn chat này.

&nbsp;