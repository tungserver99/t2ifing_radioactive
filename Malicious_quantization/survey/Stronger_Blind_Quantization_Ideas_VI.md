# Hướng mạnh hơn cho blind quantization — 2026-09-23

Đây là đề xuất nghiên cứu, chưa được triển khai hoặc chứng minh hiệu quả. Mục tiêu
BA 0.5–0.6 phải đi cùng double-tail TPR và chất lượng từng ảnh. Không có bảo đảm
đạt mục tiêu khi không dùng extractor/key của victim.

## Bằng chứng hiện có

Đọc lại `output_attack/run_20260923_083114/suite_results.csv`:

| Phương pháp | Bit accuracy | TPR | SSIM |
|---|---:|---:|---:|
| W4 natural residual | .7617–.7719 | .62–.67 | .8203–.8232 |
| FP32 natural fine-tune | .6148–.6279 | .01–.06 | .8131–.8148 |

Ba run cùng seed và khác preservation weight, không phải ba replicate độc lập.
FP32 có ngân sách/tham số khác W4. Chênh lệch chỉ gợi ý kiểm tra khả năng biểu
diễn và objective của quantizer, không chứng minh quantization là nguyên nhân duy nhất.
Chưa có kết quả của các đối chứng science mới trong bảng này.

Code hiện tại: rounding chọn floor/ceil theo scale; scale nằm trong .8–1.25 lần
khởi tạo. Nhánh code-offset có không gian rộng hơn rounding, phải được ghi riêng.
Việc nguồn trọng số giữ nguyên không tự chứng minh mọi offset đều là PTQ thông thường.

## Nghiên cứu liên quan và phần không nên claim mới

- [Stable Signature is Unstable](https://arxiv.org/html/2405.07145v1): dùng ảnh
  natural không cần ghép với ảnh generated; fine-tune decoder bằng reconstruction,
  perceptual và adversarial loss. Paper đã là tiền lệ cho purification teacher.
- [FlexRound, ICML 2023](https://proceedings.mlr.press/v202/lee23h.html): học rounding
  qua phép chia và học grid size. Học rounding/scale đơn thuần không phải novelty.
- [QuRA, NDSS 2026](https://www.ndss-symposium.org/ndss-paper/rounding-guided-backdoor-injection-in-deep-learning-model-quantization/):
  tối ưu rounding để tạo hành vi có chủ đích; không thể suy ra trực tiếp kết quả
  của nó cho Stable Signature.
- [SmoothQuant](https://arxiv.org/abs/2211.10438) và
  [OmniQuant, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/c6483c8a68083af3383f91ee0dc6db95-Abstract-Conference.html):
  phép biến đổi tương đương và tối ưu calibration đã có tiền lệ cho LLM.
- [Controllable regeneration, ICLR 2025](https://arxiv.org/abs/2410.05470): tạo lại
  ảnh với điều khiển nội dung là hướng khác, cần thêm model/cơ chế sinh ảnh.
  Không được coi kết quả đó là bằng chứng cho quantizer-only của chúng ta.

Lượt khảo sát có mục tiêu này không xác lập quyền tuyên bố “đầu tiên”.

## 1. Ưu tiên thực nghiệm: truyền thay đổi của teacher sang quantizer

Tạo teacher T bằng fine-tune bản sao model đã watermark trên natural TRAIN;
không nạp clean original decoder, không đọc extractor/key. Chọn teacher bằng
natural SEARCH và chất lượng generated SEARCH theo lịch đã khai báo trước.
Teacher là model đã purification theo objective, không mặc nhiên sạch watermark.

Sau đó quay lại trọng số marked gốc theta_w để tạo student D_Q. Với cùng latent
z từ prompt TRAIN và cùng latent E(x) của natural TRAIN, học student tái tạo đầu
ra teacher. Có thể biểu diễn target dưới dạng thay đổi:

    d_T(z) = T(z) - D_w(z)
    d_Q(z) = D_Q(theta_w)(z) - D_w(z)
    L = E ||d_Q(z) - d_T(z)||² + beta L_natural + quality constraints.

Loss hiệu này chính là output distillation về mặt đại số khi cùng D_w; không
được quảng bá phép viết lại này như một loss mới. Nó hữu ích để phân tích hướng
thay đổi và so sánh với residual PCA, vốn chứa nhiều reconstruction error.

Vì sao đáng thử: natural reconstruction hiện huấn luyện trên E(x); evaluation
lại dùng latent tạo bởi UNet. Teacher cung cấp target cùng latent generated,
giảm một nguồn lệch phân phối. Mức lợi ích còn phải đo; teacher có thể vẫn giữ
watermark hoặc giảm texture, và student có thể không biểu diễn được nó.

Giữ theta_w cùng bias/norm ngoài scope bất biến ở student. Final phải xuất
codes/scales theo quantizer đã khai báo. Nếu khởi tạo student bằng theta_teacher
rồi RTN thì đó là đối chứng fine-tune → RTN, không phải phương pháp này.
Phân biệt threat model cấm mọi fine-tuning trung gian với threat model chỉ ràng
buộc artifact triển khai: teacher chỉ phù hợp với loại thứ hai.

Đối chứng bắt buộc: fixed W4, natural reconstruction W4, FT→RTN W4, student với
teacher marked gốc, student với purification teacher. Tổng chi phí teacher phải
được tính vào phương pháp; không chỉ báo thời gian student.

## 2. Ứng viên đóng góp mạnh hơn: tham số hóa tương đương trước lượng tử hóa

Với một linear/conv operator tại vị trí hợp lệ, đặt S là ma trận chéo dương trên
input channel:

    W h + b = (W S^-1)(S h) + b.

FP32 giữ cùng hàm, nhưng:

    Q(W S^-1)(S h) + b != Q(W)h + b, nói chung.

Tối ưu S, rounding và grid scale theo target teacher ở mục 1, dưới ràng buộc
chất lượng. Điểm nghiên cứu là quyền tự do chọn cách biểu diễn cùng một model
có thể thay đổi độ bền watermark sau quantization đến mức nào.

Tạm gọi hướng này là quantization trên các tham số hóa tương đương; tên chỉ để
trao đổi, không phải khẳng định thuật toán mới. SmoothQuant/OmniQuant là tiền lệ
trực tiếp của cơ chế đổi tham số. Đóng góp chỉ thuyết phục nếu chứng minh cơ chế
watermark-specific, không thể chỉ đổi objective rồi gọi đó là novelty.

Điều kiện triển khai quan trọng:

1. Kiểm tra FP32 trước/sau transform khớp số học, watermark không suy giảm trước quantization.
2. Chỉ áp dụng biến đổi đúng cấu trúc. Không tự ý đẩy scale qua SiLU, GroupNorm
   hoặc residual addition; chúng không nói chung giao hoán với scaling.
3. Nếu không fold được S thì giữ phép nhân activation tường minh, báo chi phí.
   Không lưu S riêng từng trọng số để che một bộ trọng số FP32 tùy ý.
4. Với quantizer hiện tại per-output-channel, scaling input-channel mới làm
   thay đổi hình học trong một hàng. Scale đều cả output-channel có thể bị
   scale quantizer triệt tiêu và không mở rộng hành vi như kỳ vọng.
5. Bound condition number/range của S, cố định bitwidth và báo precision của
   scale/activation. Đây là threat model mở rộng có equivalent transforms,
   không còn đúng nguyên xi lớp Q(theta_w) cũ.

Thiết kế factorial: S=I / S theo calibration benign / S tối ưu theo teacher,
kết hợp teacher marked / teacher purified. Giữ bitwidth, target, dữ liệu và
ngân sách nhất quán. Đánh giá hết cấu hình đã khai báo, không dùng owner TEST
để chọn S, teacher hoặc hyperparameter.

## 3. Phân tích có thể tạo đóng góp riêng: khả năng quantizer biểu diễn target

Đặt e=Q(theta_w)-theta_w và tuyến tính hóa output bằng J_theta. Kiểm tra:

    min_{e trong tập perturbation lượng tử hợp lệ}
        E ||J_theta(z)e - d_T(z)||²
        subject to quality budget.

Đây là proxy cục bộ, không phải mô hình chính xác cho perturbation lớn. Có thể
đo tỷ lệ sai số target còn lại trên calibration/SEARCH, rồi kiểm tra lại decoder
phi tuyến thật. Nghiệm từ optimizer chỉ là một nghiệm tìm được; residual lớn
không chứng minh toàn bộ không gian lượng tử không có nghiệm tốt hơn.

So sánh rounding-only, rounding+scale, code-offset bounded và equivalent
transforms bằng cùng target. Nếu target thay đổi của teacher không truyền được
qua W4 nhưng truyền được qua lớp transform mở rộng, đó là phát hiện cơ chế
cụ thể hơn “thêm loss làm TPR giảm”. Owner gradient chỉ phân tích sau freeze,
không quay lại dùng để chọn phương pháp trên cùng final test.

## Kế hoạch quyết định, không chạy sweep mù

1. Hoàn thành đối chứng science mới trước khi bỏ thêm một ladder lớn.
2. Pilot cùng dữ liệu ở 256px và 512px, rồi báo cả hai; đây là ablation lệch
   độ phân giải TRAIN/TEST, không phải resize ảnh output như một attack.
3. Teacher–student là bước thực dụng đầu tiên. Nếu teacher vẫn không tiến gần
   BA .6 ở mức chất lượng yêu cầu, không hứa distillation sẽ xuống .5.
4. Nếu teacher cải thiện nhưng student không theo, đo target error và thử
   equivalent transforms ở vài vị trí tuyến tính đã kiểm tra equivalence.
5. Sau development, freeze cách chọn bằng loss/quality không có key và chạy
   final mới. Báo histogram BA, double-tail TPR và joint quality/evasion.

Mục tiêu BA .5–.6 với SSIM >=.9 là một yêu cầu thực nghiệm mới: kết quả FP32 cũ
chỉ đạt SSIM khoảng .813–.815. Không được ngoại suy rằng chất lượng .9 đã khả thi.
BA trung bình .5 còn có thể là hỗn hợp ảnh BA=0 và BA=1, đều bị double-tail bắt.
