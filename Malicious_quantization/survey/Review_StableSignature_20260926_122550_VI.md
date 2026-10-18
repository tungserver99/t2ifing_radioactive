# Stable Signature: run 20260926_122550 và cải tiến các nhánh quantization

## Kết quả thật của run

Run hoàn tất, không có branch failure; 9.656 giây (~2 giờ 41 phút). Một seed 3407, 100 ảnh test. Baseline: BA 98.56%, TPR 99/100. Các số dưới đây lấy từ `suite_results.csv`, không lấy từ loss TRAIN.

| Nhánh W4 | Checkpoint | BA ↓ | TPR ↓ | SSIM ↑ | Joint success ↑ |
|---|---|---:|---:|---:|---:|
| Fixed PTQ | RTN | 98.35% | 99% | .940 | 0/99 |
| Reconstruction | SEARCH, bước 1500 | 98.38% | 99% | .972 | 0/99 |
| Natural rounding | SEARCH, bước 1900 | 81.12% | 82% | .866 | 16/99 |
| Natural rounding | cuối, bước 2000 | 80.56% | 80% | .866 | 17/99 |
| Residual | bước 2000 | 70.38% | 41% | .814 | 31/99 |
| Warm residual QAT | SEARCH, bước 200 | 77.54% | 76% | .847 | 21/99 |
| Warm residual QAT | cuối, bước 2000 | **69.48%** | **35%** | **.833** | **44/99** |
| Teacher rounding | SEARCH, bước 1900 | 74.50% | 56% | .842 | 31/99 |
| Teacher rounding | cuối, bước 2000 | 73.60% | 53% | .842 | 34/99 |

Joint success yêu cầu ảnh vốn bị phát hiện ở baseline vừa né detector, vừa đạt PSNR≥25 và SSIM≥.8. Warm-QAT cuối có quality pass 65/100; SSIM trung bình trên .8 không có nghĩa mọi ảnh đều đạt.

**Warm-QAT cuối có tiến triển thực nghiệm**, hơn residual về cả TPR, SSIM và joint success trên tập này. Tuy nhiên checkpoint SEARCH của nó lại kém. Đây là hạn chế của proxy selection, không phải lỗi số học của detector. Không được thay checkpoint được chọn bằng checkpoint cuối sau khi nhìn TPR rồi gọi đó là blind selection.

Các đối chứng fine-tune mạnh hơn về watermark trong run này (FP32 full TPR0%, joint→W4 TPR0%), nhưng không thuộc phạm vi phát triển lần này. Code huấn luyện, loss, learning rate và số bước của chúng được giữ nguyên. Nhánh teacher vẫn dùng teacher từ fine-tune như trước; không biến nó thành model-only quantization thuần.

## Chẩn đoán theo nhánh

- **Fixed PTQ/reconstruction:** bảo toàn watermark gần như hoàn toàn. Đây là đối chứng, không có watermark objective. Reconstruction SSIM .972 là tốt đúng nhiệm vụ của nó.
- **Natural rounding:** chỉ chọn floor/ceil sát trọng số ban đầu, chưa học scale; giới hạn biểu diễn có thể cản việc học ánh xạ natural reconstruction.
- **Residual:** cải thiện suppression, nhưng quality pass chỉ 60%. PCA residual có thể ưu tiên texture/VAE error; không được gọi basis này là ownership subspace.
- **Warm-QAT:** chỉ ~1% mã khác warm center ở cuối. SEARCH objective bước 200 là .034319, bước 2000 là .034602, chênh ~0.8%, nhưng TPR chênh 41 điểm phần trăm. Proxy không phản ánh đủ suppression. Mở rộng scale giúp chất lượng sớm, còn mã cần thời gian để vượt ô lượng tử.
- **Teacher:** gradient teacher bước 2000 ~4.93e-6, reconstruction ~8.25e-5, tức nhỏ hơn ~17 lần trên minibatch đó. Chỉ báo này không đại diện toàn bộ training, nhưng xác nhận việc cân lại objective đáng kiểm tra. TRAIN teacher-to-marked MSE ~.0016545.

## Các thay đổi đã áp dụng

Profile `transfer` bật `--quant-refinement balanced`. Giữ `legacy` để tái lập.

| Nhánh | Thay đổi |
|---|---|
| Fixed PTQ | Giữ RTN làm mốc đối chứng ổn định |
| Reconstruction | Học thêm scale theo output channel; vẫn tối ưu tái tạo model marked |
| Natural rounding | Đổi từ sigmoid floor/ceil sang mã integer STE ±2 ô quanh RTN và scale trong [0.8,1.25] lần scale tâm |
| Residual | Cùng mở rộng quantizer; dùng basis contrastive dựa trên covariance residual và texture TRAIN |
| Warm residual QAT | Import đầy đủ codes **và scales** đã chọn từ residual; phạt offset quanh tâm mới, tiếp tục học cả hai |
| Teacher rounding | Cùng mở rộng quantizer; chuẩn hóa distillation bằng teacher-to-marked MSE trên TRAIN |

Các thay đổi chung cho bốn nhánh attack trên:

- Phạt chất lượng từng ảnh trên preservation TRAIN và trong SEARCH objective, trọng số .002. Warm cũ dùng .01; không áp chỉnh sửa này vào fine-tune.
- SEARCH chỉ chọn trong nửa sau ngân sách update; nếu chưa tới mốc, giữ bước 0 làm fallback. Đây là burn-in predeclared, không phải chọn theo TPR. Final checkpoint vẫn được đánh giá riêng khi khác checkpoint SEARCH.
- LR mã .01, LR scale .001, warmup/cosine. Giữ 2000 bước, không đơn thuần tăng thời gian chạy.
- Activation checkpointing cho các nhánh được cải tiến để hạn chế bộ nhớ. Có thể tăng thời gian mỗi bước; chưa có benchmark GPU.
- Ghi `quant_refinement`, `selected_step`, `quant_selection_start`, `quant_quality_weight`, `effective_teacher_weight` vào CSV suite; ghi tỷ lệ thay mã/scale và quality penalty vào SEARCH.

Teacher loss mới có hệ số hiệu dụng:

\[
 \lambda_T=\frac{0.005}{\max(\mathbb E_{TRAIN}\|T(z)-D_w(z)\|^2,10^{-4})}.
\]

Với tín hiệu TRAIN của run cũ, hệ số tương ứng ~3.02 thay vì 1. Không dùng SEARCH hoặc owner để đặt hệ số; target teacher vẫn frozen. Near-identity teacher vẫn cần đọc cảnh báo, không chứng minh nguồn target sạch.

Contrastive basis dùng cơ chế đã có trong repo, không phải đóng góp mới đã được chứng minh. Nó ưu tiên tỷ lệ residual/texture theo thống kê TRAIN; có thể vẫn chọn lỗi tái tạo tự nhiên. Cần ablation `--residual-basis pca` trước khi kết luận tốt hơn PCA.

Scale warm-QAT giới hạn quanh **scale của residual**, nên phạm vi so với model gốc có thể rộng hơn [0.8,1.25]. Checkpoint vẫn có codes W4 và scale hợp lệ; không claim phạm vi biến đổi giống bản cũ hoặc kernel INT4.

## Phạm vi và tính hợp lệ

Đổi cách biểu diễn quantizer/loss không bảo đảm TPR sẽ giảm. Tập test này đã được xem để định hướng phát triển nên là development benchmark; kết quả bài báo cần prompt/key/seed held-out mới. Không sử dụng key/extractor trong gradient hay chọn checkpoint. Không đổi nhánh fine-tune và không thêm fine-tune mới.

## Chạy

```bash
git pull origin main
# Chỉ Stable Signature
bash run_blind_quantization.sh --profile transfer

# Hoặc chạy cả hai watermark như trước
bash run_blind_quantization.sh
```

Để ablation, các cờ lower-level gồm `--quant-refinement legacy`, `--residual-basis pca`, `--quant-selection-start 0`. `reconstruction-learn-scale` là cờ riêng. Cần lưu toàn bộ command/manifest khi so phiên bản; tên nhánh giữ nguyên nhưng không gian tham số đã đổi.
