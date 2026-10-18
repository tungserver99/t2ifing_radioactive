# Kiểm tra suite coherent và cải tiến GA — 27/09/2026

Suite: `sleeper_parallel_20260927_092610_703953`. Đây là báo cáo tạm thời:
CFG đã hoàn tất owner evaluation/FID, ba nhánh còn lại đã train 2.000 bước
nhưng vẫn đang đánh giá. Chưa có kết quả GA; hàng đợi đang đợi suite này.

## Kết quả đã đo

100 ảnh có trigger, 48 bit, double-tail: phát hiện nếu khớp >=36 hoặc <=12 bit.
Baseline BA 99,5833%, TPR 100/100: baseline hợp lệ.

| Chỉ số CFG reconstruction W4 | Kết quả |
|---|---:|
| Bit accuracy | 98,6667% |
| TPR | 100% (100/100) |
| Khoảng tin cậy 95% TPR | 96,30–100% |
| SSIM triggered | 0,73237 |
| PSNR triggered | 20,744 dB |
| FID triggered tới marked reference | 80,810 |
| SSIM ordinary | 0,74749 |
| FID ordinary tới marked reference | 78,424 |
| Tỉ lệ ảnh triggered đạt quality gate | 14% |
| Joint success | 0/100 |

Kết luận: nhánh CFG chưa tạo evasion trong mẫu test. Chất lượng ảnh cũng chưa
đạt yêu cầu ở phần lớn mẫu. So run cũ có code LR cao hơn (TPR 98%, SSIM 0,621),
run này giữ ảnh tốt hơn nhưng watermark vẫn được phát hiện toàn bộ. Hai run
khác tham số nên không coi là một đối chứng thuật toán hoàn toàn tương đương.
FID với 100 ảnh chỉ mang tính thăm dò.

## Chẩn đoán coherent sau training

Các thống kê sau tính trên **72 thời điểm được ghi log**, không phải mọi bước:

| Nhánh | Median correction RMS | Max correction RMS | Tỉ lệ correction bằng 0 |
|---|---:|---:|---:|
| coherent W4 | 1,18e-7 | 1,26e-5 | 47,22% |
| delta coherent | 5,10e-7 | 1,40e-5 | 43,06% |

Delta coherent cuối run chỉ có khoảng 0,000690% delta codes khác 0 và delta
weight RMS 1,08e-6. Đây là dấu hiệu can thiệp rất yếu. Không thể suy TPR từ
các giá trị đó, nhưng có đủ lý do không chỉ tăng thêm training steps hoặc
nhân hiệu chỉnh lên một hệ số lớn tùy ý. Chưa chứng minh proxy đo ownership.

## Thay đổi tiếp theo

Giữ nguyên process đang đánh giá. Cải tiến hai nhánh GA/random còn chờ:

1. Loại chromosome trùng trước khi tính fitness; giữ ngân sách 156 chromosome
   khác nhau cho cả GA và random. Có kiểm tra dung lượng không gian tìm kiếm
   và fallback hữu hạn khi gần vét hết không gian.
2. Giữ gate loss trung bình và thêm gate theo từng record SELECT so với RTN.
   Điều này tránh tình huống vài ảnh hỏng nặng bị trung bình che đi. Không dùng
   owner score, không đổi ngưỡng detector, không dừng toàn bộ run khi một
   candidate không đạt. CSV/JSON vẫn lưu candidate bị loại.

Đây là cải tiến hiệu suất tìm kiếm và tính tin cậy của bước chọn, **chưa phải
bằng chứng tăng sức tấn công**. Điểm chưa giải quyết là proxy có thực sự tác
động watermark hay không. Chờ owner evaluation của coherent và GA/random
để quyết định hướng tiếp theo, thay vì liên tục bổ sung nhánh chưa kiểm chứng.
