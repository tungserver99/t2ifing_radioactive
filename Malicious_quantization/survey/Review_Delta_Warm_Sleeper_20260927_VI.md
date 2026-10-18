# Review ngày 27/09/2026: checkpoint, SleeperMark và quantized delta

## 1. Kết quả SleeperMark thực tế

Nguồn: `output_attack/sleepermark_20260926_150848_766174/`:
`watermark_retention.csv`, `ordinary_fid.csv`, `fid.csv`, training CSV.
100 ảnh, detector double-tail 48 bit: phát hiện khi số bit khớp ≥36 hoặc ≤12,
FPR mục tiêu tổng 0.001. Baseline phát hiện 100/100.

| Nhánh | Bit acc | TPR | SSIM ordinary | SSIM triggered | Joint success | FID ordinary / triggered |
|---|---:|---:|---:|---:|---:|---:|
| Marked | 99.58% | 100% | 1.000 | 1.000 | 0/100 | 0 / 0 |
| Fixed PTQ | 99.33% | 100% | 0.682 | 0.679 | 0/100 | 98.03 / 101.04 |
| CFG reconstruction | 98.50% | 100% | 0.736 | 0.719 | 0/100 | 78.86 / 88.82 |
| Equivariance | 98.35% | 99% | 0.733 | 0.720 | 0/100 | 82.22 / 87.71 |
| Adversarial equivariance | 98.40% | 100% | 0.733 | 0.720 | 0/100 | 81.90 / 88.26 |

Đây là thất bại của proxy hiện tại trong run đã đo, không phải suppression mạnh.
Một ảnh thoát detector ở equivariance không vượt điều kiện chất lượng cùng ảnh.
FID trên 100 ảnh so với marked reference là phép đo lệch phân phối tham khảo,
không phải FID chuẩn với tập ảnh tự nhiên lớn. Không đổi nhãn thất bại thành
thành công chỉ vì dùng FID thay SSIM.

CFG reconstruction cải thiện chất lượng so với RTN, nhưng vẫn giữ watermark.
PGD context thêm chi phí mà không cải thiện hiệu quả trong run này. Giữ code
`adversarial_equivariance_qat`, bỏ khỏi mặc định; giữ equivariance làm đối chứng.

## 2. Vì sao warm-QAT chọn checkpoint kém hơn endpoint?

Run Stable Signature `run_20260926_122550`: SEARCH chọn warm step 200,
BA 77.54%, TPR 76%, SSIM 0.847, joint success 21/99. Endpoint step 2000:
BA 69.48%, TPR 35%, SSIM 0.833, joint success 44/99.
SEARCH objective lần lượt khoảng 0.034319 và 0.034602: proxy ưu tiên điểm
không phải điểm watermark tốt nhất. Đây là lệch objective, không phải mất checkpoint.

Sửa: `transfer` dùng `--warm-checkpoint-policy final` từ trước khi chạy TEST.
Luôn xuất bước cuối, vẫn lưu `proxy_preferred_candidate`, SEARCH CSV và quality
flags. Không dừng khi endpoint không đạt chất lượng, cũng không gọi nó là
successful attack khi quality fail. `search` vẫn là tùy chọn để đối chiếu.

Không thể bảo đảm lấy checkpoint có TPR thấp nhất mà không có feedback watermark.
Chọn theo owner TEST rồi gọi kết quả đó là blind sẽ sai protocol. Chính sách mới
được định ra từ kết quả development cũ; đánh giá xác nhận cần prompt/key mới.

## 3. DeltaZip giúp gì và không giúp gì?

[DeltaZip, EuroSys 2025](https://arxiv.org/html/2312.05215v3) nghiên cứu phục vụ
nhiều LLM fine-tuned: giữ shared base, nén delta bằng sparsity và quantization.
Thực nghiệm cho thấy delta có biên độ nhỏ, thuận lợi cho nén. Đây không phải
định lý delta luôn có variance thấp, cũng không phải công trình xóa watermark.

Nếu dùng checkpoint sạch trước watermark làm base, ta đã thay đổi giả định
chỉ có marked model. Nén delta watermark so với base sạch thậm chí có thể
bảo toàn watermark tốt hơn. Không thể suy ra giảm BA từ kết quả DeltaZip.

Nhánh mới lấy cảm hứng từ cách biểu diễn, không tái hiện thuật toán ΔCompress:

\[
W(\psi)=W_{marked}+s\,\mathrm{round}(u),\quad
\mathrm{round}(u)\in[-8,7],\quad
s_c={0.05\,\mathrm{RMS}(W_{marked,c})\over7}\exp(\ell_c).
\]

Giữ `W_marked` bất biến; chỉ học tọa độ code và scale bằng STE, scale ratio
chặn trong [0.8, 1.25]. Delta bắt đầu bằng 0 nên forward khởi đầu đúng baseline,
không chịu lỗi RTN toàn bộ trọng số ngay từ đầu. Radius 0.05 là giả thuyết
cần ablate, không phải optimum hay mức giới hạn chất lượng ảnh được chứng minh.
Stable Signature nhân thêm radius với clip được khai báo.

Gradient theo mã delta có thêm hệ số scale nhỏ. Nhánh delta dùng Adam epsilon
1e-12 thay vì 1e-8 để tránh làm update biến mất khi gradient nhỏ; vẫn clip
gradient/mã và kiểm tra finite. Epsilon các nhánh W4 cũ không đổi.

**Đổi lớp can thiệp:** FP32 base + delta4 không phải whole-model W4, không giảm
tổng lưu trữ xuống 4 bit/trọng số, không có kernel INT4. Đây là học adapter
lượng tử hóa trên base đóng băng, không phải fine-tune FP32 rồi nén delta.
Không so trực tiếp như hai thuật toán PTQ có cùng ngân sách biểu diễn.

## 4. Hai nhánh mới

- `natural_delta_residual` (Stable Signature): giữ loss natural reconstruction,
  perceptual, residual contrastive và bảo toàn ảnh của nhánh residual; thay
  cách biểu diễn bằng marked base + delta. Xuất base, mã integer và scale,
  kiểm tra materialized weights; comparison group `fp32base_delta4...` riêng.
- `delta_cfg_equivariance` (SleeperMark): can thiệp đúng các weight matrix UNet
  đã chọn. Bảo toàn ordinary CFG prediction trên trajectory TRAIN; auxiliary
  late-time lấy aligned spatial target **sau CFG 7.5**, cùng cách inference.
  Có cap RMS correction, loại border, không xử lý resize/JPEG ảnh đầu ra.
  Xuất delta codes/scales với hash marked base và checkpoint dequantized;
  ghi nonzero-code fraction, delta RMS, thời gian và peak VRAM.

[SleeperMark](https://arxiv.org/html/2412.04852v2) dùng latent residual và trigger
để kích hoạt watermark. Ordinary spatial inconsistency không mặc nhiên là
ownership signal. Delta có thể giảm mất chất lượng, nhưng vẫn có thể giữ TPR
gần 100% nếu TRAIN chưa chạm hành vi kích hoạt. Đây là giới hạn chưa giải quyết,
không được gọi nhánh mới là đã tìm ra trigger hay phương pháp đột phá đã chứng minh.

Thí nghiệm tiếp: so Pareto BA/TPR–quality, joint success, runtime; ablate delta
radius và CFG spatial objective độc lập trên development. Muốn xác nhận lợi ích
đến từ biểu diễn hay loss, cần giữ loss và ngân sách giống nhau. Hai thay đổi ở
SleeperMark hiện là exploratory branch, không đủ để kết luận cơ chế riêng lẻ.

## 5. Mặc định và kiểm chứng

`bash run_blind_quantization.sh` vẫn chạy Stable Signature → SleeperMark và owner
evaluation/FID. `transfer` chỉ còn fixed, reconstruction, natural, residual,
warm-QAT và delta residual; không full/joint FT, không FT→RTN, không purified
teacher. Code FT vẫn giữ để chạy chủ động. Seed 3407, preservation 0.5, 2000 bước.

Kết quả JSON/CSV ở `output_attack`; ảnh mặc định ở `output_artifacts/images`
(có thể đổi qua `WMQ_IMAGE_OUTPUT_ROOT`); model, integer codes,
delta và checkpoint ở `output_artifacts`. Các kiểm thử CPU gồm gradient thật,
identity khi delta=0, immutable base, xuất–nạp lại, warm fixed endpoint dù SEARCH
ưu tiên bước sớm, và UNet diffusers nhỏ với activation checkpointing. Không có
benchmark GPU hoặc bằng chứng giảm watermark cho các nhánh mới ở thời điểm sửa.
