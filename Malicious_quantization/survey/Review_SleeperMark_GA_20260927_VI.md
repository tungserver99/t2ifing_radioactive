# SleeperMark: GA chưa vượt RTN/random

Run hoàn tất: `sleeper_parallel_20260927_122048_887672`.
Baseline: BA 99,5833%, TPR 100/100. Detector double-tail, 48 bit,
ngưỡng inclusive <=12 hoặc >=36, FPR tổng mục tiêu 0,001.

| Nhánh | BA | TPR | SSIM triggered | FID triggered | Joint success |
|---|---:|---:|---:|---:|---:|
| Genetic W4 | 99,0625% | 100/100 | 0,67939 | 94,92 | 0/100 |
| Random W4 | 99,0417% | 100/100 | 0,68841 | 88,32 | 0/100 |

FID so với ảnh marked reference, chỉ 100 ảnh nên là thống kê phát triển,
không thay thế đánh giá FID quy mô lớn. Chênh lệch BA rất nhỏ, không chứng
minh GA tốt hơn. TPR 100/100 có CI Wilson 95% khoảng [96,30%, 100%].

## Nguyên nhân thấy được từ selection

- Random chọn candidate 0, tức RTN anchor. Chỉ 1/12 ứng viên SELECT đạt gate.
- GA chọn candidate 147, cả 12/12 shortlist đạt gate. Proxy SELECT giảm từ
  0,0014564164 xuống 0,0014178217 (~2,65%); ordinary noise loss tăng ~2,60%.
- Gate hiện bảo vệ noise prediction **so với RTN**, không bảo đảm SSIM ảnh.
  RTN vốn đã làm chất lượng giảm mạnh. `feasible=True` không có nghĩa ảnh
  đạt SSIM 0,8 hoặc watermark đã yếu đi.
- Train/search khoảng 628 giây, tổng nhánh khoảng 3644 giây. Không nên quy
  toàn bộ thời gian cho GA: tổng còn calibration, generation và owner/FID.

## Sửa code sau run này

1. Spatial-MSE GA trước đây bỏ qua `--spatial-loss-mode`. Nay dùng cùng
   `spatial_reconstruction_loss` với QAT: mặc định `noise`, hiệu chỉnh theo
   Jacobian chuyển prediction sang x0 và CFG scale. Điều này tránh timestep
   có hệ số khuếch đại x0 lớn chi phối loss chỉ vì tham số hóa. Chạy
   `--spatial-loss-mode x0` để giữ objective cũ. Đây là thay đổi objective;
   không trộn kết quả hai mode như cùng một cấu hình.
2. Reuse unconditional prediction của chính candidate trong cùng record;
   late record giảm từ 4 xuống 3 UNet forward. Không reuse prediction teacher
   hoặc prediction từ candidate khác. Không đổi precision/batch size.
3. Thêm `selection_diagnostics` ghi rõ fallback RTN, số feasible, proxy gain,
   ordinary ratio và `image_quality_guaranteed=False`.

Residual-subspace giữ nguyên chuẩn hóa energy từ FIT, không áp thêm noise
normalization. Quality-only giữ objective ordinary loss. Tất cả nhánh GA
được giảm cùng lượt forward trùng; không so ngân sách forward với revision
cũ mà bỏ qua thay đổi này.

Suite `sleeper_parallel_20260927_132212_738087` đã khởi động ở revision
`2398e28`, đang kiểm chứng adaptive/spatial, learned subspace, random subspace
và quality-only. Không sửa tiến trình đang chạy. Chưa có kết quả owner của
suite này ở thời điểm viết; không tăng generations hay khẳng định proxy
đã đúng trước khi đọc các đối chứng đó.

Kiểm chứng tiếp theo cần ưu tiên: learned-vs-random subspace và quality-only;
đo capture trên SELECT thay vì chỉ FIT. Nếu không có lợi ích riêng thì bỏ
giả thuyết proxy này. Một hướng kế tiếp là đánh giá chất lượng full rollout
trên SELECT độc lập trước khi freeze; chưa được triển khai trong thay đổi này.
