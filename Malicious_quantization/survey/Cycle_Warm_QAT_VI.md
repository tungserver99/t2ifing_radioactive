# Cycle warm-QAT: giả thuyết cần kiểm chứng

Nhánh `natural_residual_cycle_qat_warm` giữ mục tiêu tái tạo ảnh tự nhiên và residual
projection, đồng thời thêm ràng buộc latent:

L = L_warm_QAT + gamma * mean(||E(clamp(D_Q(z))) - z||² / max(mean(z²), 1e-4)).

Ở đây z là posterior mode E(x) đã cache, chưa nhân diffusion scaling_factor.
Encoder E cố định. TRAIN và SEARCH dùng cùng chuẩn hóa; gamma mặc định 0.01,
không được chọn bằng kết quả owner TEST. Trọng số gốc, bias và norm giữ nguyên;
chỉ code offset trong miền lượng tử được học, cùng giới hạn dịch mã của warm-QAT.

Ý tưởng: bảo toàn latent có thể cho phép decoder thay đổi chi tiết đầu ra mà vẫn giữ
nội dung mã hóa. Đây chỉ là giả thuyết: encoder cũng có thể giữ dấu watermark,
hoặc mô hình lợi dụng những thay đổi mà encoder ít nhạy. Loss này không phải
ownership gradient và không chứng minh residual là tín hiệu watermark.

Paper [Stable Signature is Unstable](https://arxiv.org/abs/2405.07145) là tiền lệ
fine-tune decoder để loại watermark. Nhánh này vẫn tối ưu trong miền quantizer;
không tuyên bố paper đó đề xuất cycle loss này hoặc phương pháp này là đầu tiên.

Đối chứng trực tiếp là `natural_residual_qat_warm`, cùng nguồn khởi tạo và số bước.
Đặt `--cycle-weight 0` cho kiểm thử tương đương objective; muốn so nhiều trọng số
phải khai báo sweep trước và dùng SEARCH, không chọn theo owner TEST.
Đánh giá TPR double-tail, bit accuracy, SSIM/PSNR, tỷ lệ ảnh đồng thời thoát detector
và đạt chất lượng, cùng thời gian và bộ nhớ. Cycle loss giảm riêng lẻ không đủ.
Nếu không cải thiện tradeoff trên nhiều seed/key, loại khỏi cấu hình mặc định.

Profile transfer/science thêm một nhánh W4, không tăng bước mọi nhánh.
Warm initialization cần tính thêm chi phí residual nguồn. Encoder backward tăng
bộ nhớ/time; chưa có kiểm chứng hiệu năng H200 tại máy phát triển.
