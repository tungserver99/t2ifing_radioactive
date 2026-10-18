# Sửa điểm yếu tổng quát hóa của residual subspace

## Bằng chứng hiện có (chưa phải kết quả attack cuối)

Suite `sleeper_parallel_20260927_132212_738087`, revision `2398e28`, đã xong
search; owner evaluation đang chạy tại thời điểm kiểm tra.

| Nhánh | Proxy SELECT giảm so RTN | Ordinary loss / RTN | FIT capture | SELECT capture |
|---|---:|---:|---:|---:|
| Adaptive spatial | 4,79% | 0,9878 | — | — |
| Learned global subspace | 4,10% | 0,9770 | 45,742% | 0,02356% |
| Random global subspace | 0,396% | 1,0000 | 0,01430% | 0,01986% |
| Quality-only | 1,588% | 0,9841 | — | — |

Mỗi nhánh có 12/12 shortlist feasible theo gate noise loss. Không có số
TPR/BA cuối cho suite này tại thời điểm viết; giảm proxy chưa chứng minh
giảm watermark. Learned basis trên SELECT chỉ nhỉnh hơn random rất ít về
capture tuyệt đối. Đây là dấu hiệu mạnh rằng basis toàn ảnh đang phụ thuộc
vào residual của các ảnh FIT, không phải một cấu trúc dùng chung đã xác lập.

## Biến thể mới: basis patch dùng chung

Thay mỗi residual 4×64×64 thành vector 16384 chiều, lấy các cửa sổ 3×3
trên bốn latent channel: mỗi patch là vector 36 chiều. Fit uncentered SVD
trên patch FIT; giữ rank 2 như cũ. Một basis dùng chung cho mọi vị trí và
prompt. SELECT chỉ dùng basis và normalization đã đóng băng từ FIT.

Loss vẫn là matching trong subspace và penalty phần ngoài subspace:

`mean_patch(||U(d-r)||² / 36 + lambda * ||(I-U^T U)d||² / 36) / FIT_energy`.

Lấy mọi valid window, stride 1, không padding. Pixel nội vùng được tham gia
nhiều cửa sổ hơn pixel biên; đây là khác biệt có chủ đích với global loss.
Không có extractor/key/trigger trong fitting hay selection. Không sửa
trọng số FP32 tự do; tham số can thiệp vẫn là quantizer W4.

Giữ nhánh global. Bật patch cho cả learned và Gaussian-random cùng rank:

```bash
bash run_blind_quantization.sh --watermark sleepermark \
  --methods subspace_genetic_w4 random_subspace_genetic_w4 \
  --evolution-subspace-layout patch3
```

`*_search.json` ghi layout, patch size, số vector FIT và feature dimension.
Vẫn cùng 156 candidate, cùng FIT/SELECT prompts, cùng budget UNet giữa hai
nhánh. Chi phí SVD/projection thay đổi, được tính trong wall time. Bản mới
reuse unconditional prediction nên so forward budget với revision cũ phải
đọc số đo thực tế.

## Giới hạn và tiêu chí quyết định

Patch giúp bỏ phụ thuộc vào tọa độ tuyệt đối, nhưng có thể học texture tự
nhiên, hoặc bỏ mất watermark toàn cục. Không coi nhiều patch chồng lấn là
nhiều ảnh độc lập. Capture SELECT tăng là kiểm tra tổng quát hóa residual,
không phải bằng chứng residual đó là watermark.

Chỉ giữ hướng nếu owner evaluation sau freeze cho lợi ích so với random
patch và đối chứng chất lượng ở mức suy giảm ảnh tương đương. Không tăng
rank/generations dựa trên TEST để rồi báo cùng TEST là xác nhận độc lập.
Đây là giả thuyết thực nghiệm, chưa phải đóng góp mới đã được chứng minh.
