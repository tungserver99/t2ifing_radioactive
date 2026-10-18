# SleeperMark: UNet W4 và đánh giá chất lượng phân phối

**Cập nhật:** mặc định mới dùng calibration trajectory và hai nhánh spatial QAT.
Xem [survey và thiết kế mới](SleeperMark_Spatial_QAT_Survey_VI.md); bảng nhánh bên dưới mô tả các nhánh legacy vẫn được giữ lại.

Nguồn chính thức: [repo SleeperMark](https://github.com/taco-group/SleeperMark),
[CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_SleeperMark_Towards_Robust_Watermark_against_Fine-Tuning_Text-to-image_Diffusion_Models_CVPR_2025_paper.pdf),
[owner evaluation](https://github.com/taco-group/SleeperMark/blob/a78789cbac92ff5a63a2ad13885cf3893a3d9c4f/Stage2/eval.py),
[Clean-FID](https://github.com/GaParmar/clean-fid).

## Can thiệp vào đâu?

Checkpoint công khai là UNet SD1.4 đã nhúng watermark. Scope mặc định
`up_attentions` lấy các weight ma trận trong attention của UNet up-blocks,
phù hợp vị trí nhúng được mô tả trong paper. Scope `all` lấy mọi weight ma trận
UNet. Không sửa ảnh bằng JPEG/resize để attack, không sửa VAE hay text encoder.
Mỗi nhánh khởi đầu từ cùng UNet fingerprint, xuất safetensors chứa các weight
đã thay đổi, kiểm tra số ma trận thay đổi >0, rồi nạp chính artifact đó để sinh ảnh.
Hash kiểm tra frozen components và artifact được ghi trong báo cáo.

W4 là trọng số được chiếu lên lưới integer 4-bit, lưu và thực thi dưới dạng
dequantized FP32. Không claim kernel INT4 hay lượng tử hóa toàn UNet khi chỉ dùng
scope `up_attentions`. Bias/norm ngoài scope giữ nguyên.

## Các nhánh

| Method | Dữ liệu/loss | Tham số được thay đổi |
|---|---|---|
| `fixed_ptq` | RTN, không train | W4 trên scope đã khai báo |
| `model_reconstruction` | Latent sinh từ chính model; giữ prediction gốc trên latent nhiễu theo timestep | Rounding STE + scale theo channel |
| `natural_rounding` | Encode ảnh tự nhiên, thêm noise, học target của scheduler + bảo toàn prediction gốc | Rounding STE + scale |
| `natural_finetune` | Natural denoising FP32, sau đó RTN W4 | Weight FP32 thuộc scope; xuất cả FP32 và W4 |
| `natural_joint_finetune` | Natural denoising ở cả FP32/W4, cộng bảo toàn prediction gốc | Weight FP32 thuộc scope qua STE; xuất cả FP32 và W4 |

Ảnh tự nhiên là unpaired, dùng empty text conditioning. Đây là objective diffusion
denoising, không phải công thức tái tạo ảnh bằng VAE của Stable Signature.
Timestep được lấy đều trong lịch train; xử lý `epsilon`, `v_prediction`, `sample`
theo scheduler. Mỗi nhánh học có mặc định 2.000 update; joint có hai forward/backward
mỗi update nên ngân sách compute lớn hơn. Batch 1, FP32 và native gradient
checkpointing là mặc định để giảm VRAM. Không bảo đảm tránh OOM trên GPU chia sẻ.

Không bê nguyên các loss residual/cycle trên VAE hoặc teacher của Stable Signature
sang UNet vì chúng cần một thiết kế và đối chứng riêng. Nhánh model reconstruction
là đối chứng bảo toàn model, không có lý do mặc định để kỳ vọng nó xóa watermark.
Natural fine-tune cũng chưa chắc thắng vì SleeperMark được thiết kế chống fine-tune.

## Blind và owner evaluation

Attack không nhận key, trigger hay extractor làm đầu vào loss/selection. Dữ liệu
natural được tải riêng. Train và test prompt tách nhau; nếu số train latent lớn hơn
số prompt train, prompt lặp với seed khác và được ghi rõ trong manifest.
Chọn checkpoint bước cuối theo lịch khai báo, đóng băng hash trước sinh test.
Owner sau đó dùng trigger công khai `*[Z]& ` cho **cả baseline và mọi attack**.
Extractor chính thức đọc latent của ảnh PNG sau VAE encode, lấy posterior sample
rồi nhân scaling factor. RNG lấy mẫu posterior cố định theo ảnh giữa các nhánh.

Đánh giá double-tail với ngưỡng integer từ FPR tổng (mặc định 0.001), ghi bit accuracy,
TPR, CI Wilson, bit match theo ảnh. Tập prompt thường được đánh giá riêng để đo
tỷ lệ phát hiện không trigger; đây không phải bảo đảm FPR trên mọi ảnh tự nhiên.
Nếu baseline TPR <0.9, vẫn xuất kết quả nhưng gắn `valid_clean_baseline=false`.
Không diễn giải trường hợp đó là attack thành công.

## FID

Clean-FID mode `clean`, InceptionV3 2048 chiều. Hai bảng:

- `fid.csv`: phân phối ảnh trigger so với model fingerprint gốc, cùng prompt/seed.
- `ordinary_fid.csv`: phân phối ảnh không trigger, đo ảnh hưởng lên utility thường.

Nếu cung cấp tập ảnh thật held-out, thêm cột `fid_to_real`. FID-to-reference thấp
nghĩa là ít lệch phân phối gốc, không chứng minh realism cao hay từng ảnh giữ nội dung.
Vẫn lưu SSIM/PSNR trên ảnh thường để giải thích tradeoff; chúng không chặn nhánh mới.
Tất cả FID chạy sau freeze, không tham gia chọn checkpoint.

100 ảnh tạo FID có bias/variance lớn; không so trực tiếp với FID dùng 30k ảnh COCO
trong paper. `small_sample_warning` dùng mốc 2048 chiều để cảnh báo, **không có nghĩa**
2048 mẫu là đủ đáng tin cho publication. Khi tăng `--test-n`, cần file `--prompts`
có đủ prompt test khác nhau và ít nhất một prompt train; nên predeclare tập held-out lớn.

## Checkpoint và tính tái lập

UNet/owner assets tải từ hai Google Drive folder được README chính thức dẫn tới.
Source extractor pin commit và SHA256; checkpoint chưa có checksum publisher công bố,
nên hash được ghi nhận lần tải đầu và kiểm tra khi dùng lại (trust-on-first-download).
Không gọi đó là checksum độc lập xác thực nguồn. Google Drive quota có thể làm tải lỗi.
Các file lớn nằm ngoài `output_attack`. Safetensors của nhánh là partial state theo
scope, phải nạp trên đúng UNet fingerprint nguồn; không phải một diffusers pipeline đầy đủ.

Mẫu lệnh mở rộng:

```bash
bash run_blind_quantization.sh --watermark sleepermark --scope all --steps 2000
bash run_blind_quantization.sh --watermark sleepermark --methods fixed_ptq natural_rounding
bash run_blind_quantization.sh --watermark sleepermark --fid-real-reference /data/coco_heldout
```

Lệnh không có tham số chạy tuần tự Stable Signature profile `transfer` rồi SleeperMark.
`--profile transfer` chạy riêng Stable Signature; `--watermark sleepermark` chạy riêng SleeperMark.
Không trộn kết quả hai watermark trong cùng một suite và không dùng TPR test để
điều chỉnh loss rồi báo lại chính tập test đó như đánh giá độc lập.
