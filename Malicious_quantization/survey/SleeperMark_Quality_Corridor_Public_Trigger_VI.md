# SleeperMark: tìm kiếm trong miền chất lượng và đối chứng trigger công khai

Ngày 30/09/2026. Kết quả lần chạy
`output_attack/sleeper_parallel_20260929_174002_709773`:
`prefix_invariance_genetic_w4` chọn lại ứng viên 0 (RTN), bit accuracy 99,04%,
TPR 100/100, triggered SSIM 0,688 và FID cùng seed 88,32. Trên FIT chỉ
15/156 genome có ordinary noise loss ≤ 1,25 lần RTN. Trong 12 ứng viên SELECT,
chỉ hai đáp ứng ràng buộc chất lượng; cả hai không giảm proxy hơn RTN. Đây là
lỗi phân bổ ngân sách tìm kiếm: Pareto ranking và crowding ưu tiên nhiều cực
trị proxy không thể dùng.

Áp dụng riêng cách lập shortlist mới lên 156 genome **đã đo** của run cũ
(không chạy lại model) cho 12/12 ứng viên đạt giới hạn FIT, thay vì 2/12.
Đây chỉ chứng minh cách chọn FIT hợp lý hơn; chúng chưa được đo trên SELECT,
nên không suy ra attack đã cải thiện.

## Thay đổi được kiểm nghiệm tiếp

Các nhánh evolutionary W4 hiện khởi tạo gần RTN và xếp ứng viên FIT theo thứ
tự: đạt giới hạn ordinary loss trước, proxy loss thấp sau. Ứng viên không đạt
giới hạn được xếp theo mức vi phạm. Immigrant phần lớn là perturbation thưa;
20% vẫn thăm dò toàn miền. SELECT vẫn dùng prompt riêng và kiểm tra cả trung
bình lẫn tệ nhất từng record. Shortlist gồm RTN, các genome FIT hợp lệ có
proxy tốt và một vài genome có ordinary loss thấp. Tổng số genome và số lần
đo UNet không đổi. `--evolution-legacy-search` giữ nguyên search cũ để làm
ablation. Các nhánh local/LLM nhận cùng giới hạn FIT, nhưng LLM vẫn chỉ là
generator ứng viên và không nhận owner score.

Một đối chứng mới, `public_trigger_genetic_w4`, dùng trigger `*[Z]& ` được
công bố trong [paper SleeperMark, CVPR 2025, Sec. 4.1](https://arxiv.org/html/2412.04852v2).
Trên latent/timestep của TRAIN, nó kéo dự đoán khi có trigger của UNet W4 về
dự đoán prompt gốc của model đã đánh dấu. Đây là mục tiêu khôi phục phản ứng
không trigger; ràng buộc ordinary prediction trên SELECT vẫn áp dụng. Nhánh
này **khác threat model** với các nhánh model-only blind: nó dùng side information
là trigger công khai, nhưng không dùng key, extractor hoặc kết quả owner TEST
để tối ưu/chọn checkpoint. Báo cáo và manifest ghi rõ cờ
`public_trigger_used_for_fitness`/`public_trigger_used_for_selection`.

`public_trigger_consistency_qat` là đối chứng gradient: trong mỗi bước QAT W4,
nó giữ ordinary CFG noise prediction theo marked teacher và kéo prediction của
prompt có trigger về prediction của prompt gốc. Text context được cache theo
prompt, không huấn luyện text encoder hoặc trọng số FP32 tự do. Đây cũng là
nhánh **public-trigger**, không được gộp số liệu với model-only blind. Hai nhánh
public-trigger cho phép tách lỗi proxy blind khỏi giới hạn do tìm kiếm W4 thô:
nếu QAT thành công còn GA không, độ phân giải/optimizer của GA là nghi vấn.

## Thí nghiệm phân biệt nguyên nhân

```bash
bash run_blind_quantization.sh --watermark sleepermark \
  --methods adaptive_genetic_w4 prefix_invariance_genetic_w4 public_trigger_genetic_w4
```

Chạy QAT công khai riêng vì 2.000 bước mặc định có chi phí lớn hơn 156 genome GA:

```bash
bash run_blind_quantization.sh --watermark sleepermark \
  --methods public_trigger_consistency_qat
```

Ba nhánh cùng W4, FIT/SELECT, seed, số genome và giao thức owner sau freeze.
Nếu nhánh public trigger tốt hơn nhiều mà hai nhánh blind không tốt hơn, thiếu
tín hiệu kích hoạt là nút thắt. Nếu cả ba vẫn giữ TPR cao, giới hạn có thể là
không gian W4 scale/rounding của các up-attention hoặc ràng buộc chất lượng.
Không suy ra điều nào trước khi có owner result. FID với 100 ảnh có cảnh báo
mẫu nhỏ; dùng thêm bit accuracy, TPR, SSIM và tỷ lệ đạt cả chất lượng lẫn evasion.
