# Fine-tune đồng thời FP32/W4 với cycle latent

## Câu hỏi thực nghiệm

`natural_joint_finetune` thử giảm khoảng cách giữa decoder đã fine-tune và bản W4
của chính nó. Fine-tune thuần chỉ tối ưu đầu ra FP32. Nhánh này tối ưu cả đầu ra W4
trong quá trình học, thay vì chờ fine-tune xong mới lượng tử hóa.
Đây là thử nghiệm QAT áp vào natural reconstruction, không tuyên bố novelty hay
đảm bảo bit accuracy/TPR thấp hơn. Fine-tune thuần đã có thể giảm mạnh TPR trong run
trước; mục tiêu đáng đo là chất lượng tốt hơn ở cùng mức giảm watermark sau W4.

## Phương pháp

Với ảnh natural TRAIN bất kỳ x, cache z = mode(E(x)). Không cần ảnh sạch ghép cặp
với ảnh sinh có watermark; encoder cố định, không dùng secret/extractor.

L = R(D_theta(z), x) + beta R(D_Q4(theta)(z), x)
  + gamma [C(D_theta(z), z) + beta C(D_Q4(theta)(z), z)].

R gồm MSE và LPIPS với trọng số hiện hành. C là sai số encode lại ảnh so với raw
latent z, chia năng lượng latent từng ảnh, chặn mẫu số dưới 1e-4. Beta mặc định 1,
gamma mặc định 0.01. Khi bật ft_preserve_weight, thêm generated lowpass loss cho
cả hai bản. Dual budget hiện áp trên bản FP32; quality gate SEARCH kiểm tra cả hai.

Tất cả decoder parameters, kể cả bias/norm, được fine-tune. Nhánh W4 chỉ lượng tử hóa
weight thuộc scope đã khai báo, cùng quy tắc per-output-channel và half-up rounding
với export RTN. Scale được tính lại từ trọng số hiện tại và detach mỗi bước;
STE truyền gradient về trọng số FP32. Đây là mô phỏng W4, không phải benchmark kernel INT4.

## Chọn checkpoint và kết quả

Chọn checkpoint bằng tổng objective natural SEARCH của hai bản, với quality gate
generated SEARCH cho cả FP32 và W4 khi policy=constrained. Khi không có checkpoint
đạt gate, vẫn lưu bản fallback và báo không đạt; không dừng các nhánh khác.
Không dùng TEST hay owner metrics để chọn. Giữ đối chứng natural_full_finetune.

- `natural_joint_finetune_fp32_test`: checkpoint được chọn.
- `natural_joint_finetune_rtn_w4_c1.0_test`: RTN W4 từ đúng checkpoint đó, không train thêm.
- Có thể có endpoint FP32 cuối lịch học nếu checkpoint chọn khác endpoint, như các nhánh cũ.

FP32 và W4 đều nằm ngoài threat model quantizer-only vì trọng số, bias/norm đã đổi.
Teacher của nhánh natural_teacher_rounding vẫn lấy từ đối chứng fine-tune cũ;
không tự đổi teacher bằng kết quả TEST của nhánh mới.

Các trường `joint_w4_loss`, `joint_w4_gradient_norm`, `joint_w4_psnr`,
`joint_w4_ssim`, `joint_w4_feasible` giúp kiểm tra huấn luyện/SEARCH.
Trường psnr/ssim chính của nhánh FP32 vẫn là FP32; feasible là giao hai quality gates.
Kết quả W4 được ghi riêng để tránh lẫn với số FP32.

## Chạy và ablation

```bash
git pull origin main
bash run_blind_suite.sh --profile transfer
```

Profile science cũng có nhánh này. Chỉ thêm một quá trình train 2.000 bước và một
bản W4 xuất sau train; không bật W8. Model/checkpoint/ảnh vẫn lưu ngoài output_attack.

Ở launcher trực tiếp, `--joint-quant-weight 0` bỏ gradient loss W4, nhưng SEARCH vẫn
kiểm tra cả hai quality gate; đây không hoàn toàn tương đương fine-tune cũ.
`--cycle-weight 0` bỏ cycle loss, hữu ích để tách tác dụng QAT và cycle.
Không sweep rồi chọn bằng owner TEST. Để claim đóng góp cần so các ablation trên
seed/checkpoint mới, cùng chất lượng và báo cả thời gian/bộ nhớ.

Hai decoder backward và hai encoder backward khiến nhánh mới nặng hơn fine-tune
thuần. Chưa benchmark GPU tại máy phát triển; không đảm bảo thời gian hoặc không OOM.
