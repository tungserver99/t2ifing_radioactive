# Kết quả LLM và phép thử tiếp theo cho SleeperMark

Ngày phân tích: 30/09/2026. Nguồn: `output_attack/sleeper_parallel_20260927_162600_499085`.
Ba nhánh dùng cùng seed, FIT/SELECT, W4, 156 genome và 7.942 lần gọi UNet cho
FIT/SELECT. Owner chỉ đánh giá sau khi đóng băng genome đã chọn.

| Nhánh | SELECT proxy giảm so RTN | Bit accuracy | TPR (100 ảnh) | Triggered SSIM | Triggered FID | Thời gian | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adaptive GA | 1,35% | 99,08% | 100% | 0,682 | 94,03 | 3.692 s | 4,72 GiB |
| Local proposal + GA | 2,35% | 99,00% | 100% | 0,699 | 86,98 | 3.694 s | 4,72 GiB |
| LLM proposal + GA | 2,63% | 98,94% | 100% | 0,691 | 93,41 | 3.720 s | 10,52 GiB |

Clean bit accuracy là 99,58%, TPR 100%. Cả ba nhánh vẫn giữ 100/100 ảnh bị
phát hiện và joint success bằng 0. Khoảng tin cậy 95% của TPR 100/100 là
96,30–100%. Ngưỡng SSIM báo cáo là 0,8; chỉ 0%, 10%, 8% ảnh triggered của ba
nhánh tương ứng vượt ngưỡng. FID cùng seed và SSIM cho thấy chất lượng giảm
đáng kể. Đây không phải attack thành công.

Detector bắt ảnh nếu có ít nhất 36/48 bit khớp (hoặc nhiều nhất 12/48 bit
khớp ở đuôi đảo). Ảnh yếu nhất của GA, local và LLM vẫn lần lượt có 41, 39,
40 bit khớp; trung bình là 47,56, 47,52, 47,49 bit. Vì vậy TPR 100% không
chỉ là trường hợp sát ngưỡng: hầu hết ảnh vẫn giữ gần nguyên thông điệp.

LLM chỉ đóng góp 8 genome mới trong 36 lượt đề xuất được yêu cầu; 28 lượt trùng
archive hoặc trùng nhau. Local proposer đóng góp đủ 36. LLM còn cần thêm 5,8 GiB
peak VRAM và khoảng 87 giây train so với GA (tổng wall time hơn 28 giây). Nhánh
LLM chạy ở commit `11029a2`, trước thay đổi reflection `9e303ae`. Reflection có
thể tăng số đề xuất hợp lệ; chưa có số liệu cho thấy nó cải thiện owner metric.

Điểm nghẽn chính là proxy: tối ưu mất mát ảnh/latent dưới prefix punctuation
không làm giảm thông điệp được phát hiện. Trong SleeperMark, watermark được
kích hoạt bởi text condition và nhúng qua attention ở up blocks; paper gốc ghi
rõ điều này ([SleeperMark, CVPR 2025](https://arxiv.org/html/2412.04852v2)).
Vì thế phép thử kế tiếp thay spatial-shift proxy bằng **prefix invariance**:
trên từng latent và timestep muộn của TRAIN, lấy vài prefix punctuation ngẫu
nhiên, chọn prefix làm thay đổi thành phần high-pass của dự đoán x0 nhiều nhất
trên model đã đánh dấu, rồi tối ưu quantizer W4 để giảm hiệu ứng đó. Dự đoán
noise cho prompt gốc tiếp tục bị ràng buộc trên SELECT. Việc chọn prefix không
dùng trigger, key, extractor hay TEST của owner.

Đây là kiểm tra giả thuyết về khả năng chuyển từ các prefix công khai sang
trigger bí mật, không phải ước lượng watermark. Nếu TPR và bit accuracy vẫn gần
baseline, cần bác bỏ giả thuyết chuyển đó và dừng việc tăng ngân sách search
cho proxy này. Vì cùng owner TEST đã được đọc nhiều lần trong quá trình phát
triển, kết quả hiện tại chỉ là development evaluation; bài báo cần tập đánh giá
độc lập sau khi chốt phương pháp.
