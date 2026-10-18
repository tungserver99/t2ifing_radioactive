# Kết quả cuối structured GA và sửa calibration

Suite: `sleeper_parallel_20260927_132212_738087`, revision `2398e28`.
Exit 0, FID hoàn tất cả bốn nhánh. Baseline BA 99,5833%, TPR 100/100.

| Nhánh | BA | TPR | Triggered SSIM | Triggered FID | Joint success |
|---|---:|---:|---:|---:|---:|
| Adaptive spatial GA | 98,9792% | 100/100 | 0,69409 | 88,97 | 0/100 |
| Learned global subspace | 98,8542% | 100/100 | 0,67638 | 96,78 | 0/100 |
| Random global subspace | 99,1667% | 100/100 | 0,68514 | 91,41 | 0/100 |
| Quality-only GA | 99,0833% | 100/100 | 0,69023 | 86,62 | 0/100 |

Không có evasion ở mức detector đã định. Learned subspace có BA thấp nhất
nhưng cũng SSIM/FID kém nhất; không chứng minh loại riêng watermark. Số ảnh
FID chỉ 100, dùng cho development comparison, không coi là FID chuẩn quy mô lớn.

FIT capture 45,742% so SELECT 0,02356% xác nhận basis toàn ảnh tổng quát hóa
kém; random SELECT 0,01986%. Tăng thế hệ hoặc rank chưa có cơ sở thực nghiệm.

## Hiệu năng

Mỗi nhánh mất khoảng 7199–7231 giây wall time khi chạy bốn nhánh đồng thời;
search chiếm khoảng 1214–1227 giây. Không suy ra tổng thời gian sẽ giảm cùng
tỷ lệ với thời gian search hoặc calibration.

Code cũ sinh 256 trajectory rồi chỉ lấy 16 FIT + 16 SELECT record. Code mới
lập cùng split từ metadata trước khi inference, chỉ sinh các trajectory
chứa record cần dùng. Tối đa 32 trajectory ở cấu hình mặc định, thay vì 256.
Giữ ID trajectory gốc nên seed vẫn là `seed+i`; giữ thứ tự timestep và
prompt lặp. Không chia lại FIT/SELECT sau khi rút gọn dataset.

- Chỉ bật tối ưu này khi mọi method trong tiến trình đều thuộc nhóm evolution.
- QAT/reconstruction và tiến trình hỗn hợp vẫn giữ calibration đầy đủ.
- Đối chiếu actual captured prompt/timestep với kế hoạch; lệch sẽ báo lỗi.
- Manifest ghi ID gốc, chỉ số full/compact, số trajectory thực sinh và dự kiến.
- Không đổi loss, quantizer, số candidate, precision hoặc owner evaluation.

Kiểm thử so full/compact record gồm seed với prompt độc nhất, prompt lặp và
tập nhỏ. Test tích hợp xác nhận train_evolution dùng split đã lập sẵn.
Chưa có đo GPU end-to-end cho tốc độ bản mới.

## Run tiếp theo

Patch suite `sleeper_parallel_20260927_152313_833645` đã khởi động ở revision
`217ba87`; giữ nguyên tiến trình. Chưa có owner result lúc viết. Không đưa
global subspace vào mặc định, không khẳng định patch là bước đột phá trước
khi so learned/random và chất lượng ảnh sau đánh giá.
