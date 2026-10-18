# Kết quả SleeperMark và thử nghiệm GA có cấu trúc / residual subspace

27/09/2026. Các thay đổi trong tài liệu này áp dụng cho SleeperMark UNet.
Không tự áp timestep proxy lên VAE decoder của Stable Signature.

## Kết quả suite đã hoàn tất

`sleeper_parallel_20260927_092610_703953`, 100 triggered + 100 ordinary images,
baseline BA 99,5833%, TPR 100%. Bốn nhánh đều có joint success 0/100.

| Nhánh | BA | TPR | Triggered SSIM | Triggered FID tới marked reference |
|---|---:|---:|---:|---:|
| CFG reconstruction W4 | 98,6667% | 100% | 0,73237 | 80,8096 |
| Coherent W4 | 98,6458% | 100% | 0,73454 | 78,1726 |
| Delta equivariance | 99,2083% | 100% | 0,95568 | 14,6503 |
| Delta coherent | 99,5417% | 100% | 0,99923 | 0,3979 |

Delta là FP32 base + delta INT4; không gộp với selected-weights W4 để claim
compression. FID 100 ảnh là thăm dò. TPR 100/100 có CI95% khoảng 96,3–100%,
không phải bằng chứng xác suất evasion bằng 0 trên mọi ảnh.

CFG chỉ tối ưu preservation, không dùng spatial removal proxy. Vì vậy TPR
100% của CFG riêng lẻ không đánh giá được proxy. So sánh coherent với CFG
cho thấy chưa có lợi ích về evasion; delta coherent gần như giữ nguyên model.
Không tăng steps hay khuếch đại correction gần 0 chỉ vì muốn attack mạnh hơn.

Hai nhánh genetic/random cũ đang chạy trong
`sleeper_parallel_20260927_122048_887672` khi chuẩn bị thí nghiệm mới. Giữ
nguyên code revision của run đó để làm đối chứng.

## Ứng dụng survey đã triển khai

Nguồn và giới hạn được ghi trong
[survey Huỳnh Thị Thanh Bình](Huynh_Thi_Thanh_Binh_Evolutionary_Quantization_VI.md).

1. **Crossover theo cặp scale–bias:** giữ cặp gene của cùng attention module
   khi lai ghép. Lấy cảm hứng từ linkage của
   [MF-LTGA](https://arxiv.org/abs/2005.03090). Đây là grouping theo cấu trúc,
   chưa phải học linkage tree từ population 12 cá thể.
2. **Đa dạng prediction:** với mỗi TRAIN record, lấy sai khác CFG prediction,
   adaptive-average-pool về 2×2 và thêm RMS; chuẩn hóa bằng RMS teacher. Trong
   mỗi Pareto rank, giữ các cực trị objective rồi dùng farthest-point selection
   để chọn hành vi khác nhau. Không thêm UNet forward để tạo descriptor. Ý tưởng
   diversity tham khảo [HSEvo](https://arxiv.org/abs/2412.14995) và
   [QDEvo](https://arxiv.org/abs/2607.11916); không dùng code embeddings hoặc LLM.
3. **Operator thích nghi:** single-gene mutation, paired mutation, paired
   crossover và random immigrant. Tín dụng được cập nhật từ FIT improvement so
   với parent; offspring cải thiện ít nhất một objective được ghi nhận, còn
   survival vẫn theo toàn population. Xác suất mỗi operator ít nhất 5%.
   Lấy cảm hứng từ adaptive operators trong
   [EME-BI](https://github.com/cuonglvsoict/EME-BI_TEVC22), không claim tái tạo
   toàn bộ EME-BI/multitasking. Duplicate restart không được ghi công cho operator
   đã tạo duplicate. JSON ghi operator, xác suất và reward từng candidate.

Flags đối chứng độc lập: `--evolution-ablate-grouping`,
`--evolution-ablate-diversity`, `--evolution-ablate-adaptation`.

## Nghiên cứu và proxy mới

[SleeperMark, CVPR 2025](https://arxiv.org/html/2412.04852v2) hướng đến tách
watermark khỏi ngữ nghĩa, kích hoạt qua trigger. Điều đó giúp giải thích vì sao
chỉ giữ/đổi ordinary semantics chưa chắc làm suy watermark. Không suy ra rằng
residual không gian đo trên prompt thường chính là secret residual.

[Shallow Diffuse](https://arxiv.org/abs/2410.21088) khai thác low-dimensional
subspaces trong một cơ chế watermark khác. Nó là động cơ khảo sát cấu trúc
subspace, không phải bằng chứng basis của chúng ta chứa watermark SleeperMark.

Thay đổi đề xuất ở đây là **đo và tái tạo residual trong một subspace cố định**,
thay vì MSE toàn latent tới pseudo-target. Giữ nguyên marked teacher, random
prefix độc lập và spatial target cũ để tách đóng góp của loss.

Đặt `b_i` là teacher x0 ở augmented context và `r_i = target_i - b_i`.
Học basis trực chuẩn U bằng SVD không trừ mean từ các residual late-timestep
của FIT. Mặc định rank tối đa 2, tự giảm theo numerical rank. Basis ngẫu nhiên
cùng rank được tạo bằng Gaussian + QR làm đối chứng.

Với `d_i = student_x0 - b_i`, D là số tọa độ latent và P=U^T U:

```
E_fit = max(mean_i ||r_i||² / D, 1e-8)
L_proxy = mean_i [ ||U(d_i-r_i)||² / D
                 + 0.25 * ||(I-P)d_i||² / D ] / E_fit
```

Term đầu khớp phần residual nằm trong basis. Term thứ hai phạt thay đổi ngoài
basis, giúp hạn chế tác động lan sang cấu trúc khác. Ordinary CFG/unconditional
preservation vẫn là objective riêng, với gate trung bình và từng mẫu SELECT.
Normalization dùng FIT, giống nhau giữa learned và random basis; không chia
cho năng lượng projection ngẫu nhiên rất nhỏ rồi khuếch đại nó lên.

SELECT không được fit lại basis/rank/normalization. Report ghi rank thực,
residual RMS, năng lượng basis giải thích trên FIT và SELECT. Nếu FIT cao nhưng
SELECT thấp thì không có bằng chứng basis tổng quát. Basis rank 0 được ghi
`active=false`; loss hữu hạn, toàn thí nghiệm vẫn chạy để thu kết quả.

SVD chỉ trên ma trận số hàng nhỏ ở CPU. Population lưu gene/descriptor/metrics,
không giữ nhiều UNet trong GPU. Mỗi fitness chỉ đồng bộ một lần để chuyển loss
và descriptor về CPU. Forward vẫn dùng hard signed W4 dequantized FP32; không
fine-tune trọng số tự do và không claim kernel INT4.

## Bốn nhánh mới

| Method | Search | Proxy |
|---|---|---|
| adaptive_genetic_w4 | grouped + behavior diversity + adaptive operators | spatial MSE cũ |
| subspace_genetic_w4 | như trên | learned residual subspace |
| random_subspace_genetic_w4 | như trên | random subspace cùng rank |
| quality_genetic_w4 | như trên | quality-only, hai fitness cùng ordinary loss |

Quality-only vẫn chạy các forward phụ nhưng không dùng proxy để chọn, để giữ
cùng ngân sách forward đánh giá candidate trong đối chứng này. Có thể tối ưu
bỏ phần compute đó ở triển khai thực tế, nhưng không gộp hai cấu hình trong
cùng claim ngân sách. SVD/descriptor có overhead được tính vào wall time.

```
bash run_blind_quantization.sh --watermark sleepermark --methods adaptive_genetic_w4 subspace_genetic_w4 random_subspace_genetic_w4 quality_genetic_w4
```

Mặc định 12 cá thể × (12+1) thế hệ = 156 chromosome khác nhau/nhánh; 16 FIT
records và 16 SELECT records, tách theo prompt. Tập initial population giống
nhau khi cùng seed. Sau FIT, chốt từ shortlist theo SELECT, rồi freeze trước
khi load extractor/key và chạy owner evaluation/FID như pipeline cũ.

Kết quả nhẹ nằm trong `output_attack`; checkpoint/ảnh ở các thư mục nặng đã
cấu hình. Không tắt kiểm tra chất lượng; candidate vi phạm vẫn được lưu nhưng
không được chọn. Giữ RTN anchor làm fallback.

## Cách kết luận

- Adaptive GA so với GA cũ: cải tiến search có lợi ở cùng objective không?
- Learned subspace so với adaptive spatial: đổi proxy có lợi không?
- Learned so với random subspace và quality-only: có lợi riêng từ residual
  structure hay chỉ do preservation/nhiễu lượng tử hóa?
- So ở chất lượng ảnh tương đương và báo joint success; không chỉ đọc BA.

Đây là giả thuyết mới được triển khai để kiểm nghiệm. Chưa có kết quả để gọi
là đột phá hoặc hứa BA 0,5–0,6. Nhiều vòng đọc owner TEST để chỉnh phương pháp
biến các run hiện tại thành development evaluation; bài báo cần final holdout
mới sau khi chốt phương pháp.
