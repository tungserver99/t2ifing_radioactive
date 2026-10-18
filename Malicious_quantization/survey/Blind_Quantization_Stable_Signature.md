# Blind quantization: model-only và ảnh tự nhiên không ghép cặp

## Cập nhật 2026-09-22 sau pilot

[Revision residual quantization](Residual_Quantization_Revision_VI.md) ghi phân tích
report thật và nguồn nghiên cứu. [Revision sau run 000857](Residual_QAT_Finetune_Revision_VI.md)
thêm residual QAT và đối chứng FP32. Mặc định: fixed_ptq, reconstruction, natural_residual,
natural_residual_qat (W4) và natural_full_finetune (FP32). Vẫn giữ model-only; các nhánh
cũ gọi được tường minh. Natural residual là giả thuyết chưa xác nhận bằng owner test,
không phải ownership subspace đã biết. Phần sau mô tả objective/protocol cũ làm đối chứng.

## Hai hướng được giữ để so sánh (cập nhật 2026-09-21)

Hướng model-only bên dưới vẫn giữ nguyên mục tiêu smoothing. Launcher
tải một pool ảnh COCO công khai rồi thêm các nhánh chọn bởi `--natural-methods`
ở mỗi bits/clip. Cặp natural cũ là `natural_rounding` và `natural_rounding_scale`.
Ảnh tự nhiên không cần là ảnh sạch tương ứng với bất kỳ ảnh model sinh nào.
Đây là threat model rộng hơn model-only: có thêm dataset ngoài, vẫn không có
key, extractor, detector feedback hay checkpoint model sạch trong optimization.
Nhánh natural còn dùng LPIPS pretrained công khai làm perceptual prior; nhánh
model-only hoàn thành trước khi nạp dataset/prior này và không dùng chúng để học.

Với ảnh tự nhiên x, dùng encoder của chính marked pipeline để tính z = E_w(x)
(posterior mode, VAE-space, không nhân/chia diffusion scaling factor).
Chỉ quantizer của decoder được học:

```
L = MSE(D_Q(E_w(x)), x)
    + natural_perceptual_weight * LPIPS(D_Q(E_w(x)), x)
    + preserve_weight * MSE(D_Q(z_generated), D_w(z_generated))
```

Riêng nhánh QAT purification thay thành:

```
L_qat = MSE(D_Q(E_w(x)), x) + lambda_lpips * LPIPS
        + lambda_sem * MSE(LPF_8(D_Q(z_generated)), LPF_8(D_w(z_generated)))
        + lambda_trust * mean(code_offset^2)
```

Mỗi bước dùng một ảnh tự nhiên train và một latent sinh train. Encoder, UNet,
trọng số decoder nguồn, bias và normalization đóng băng. Nhánh thứ nhất học
rounding; nhánh thứ hai thêm per-channel scale giới hạn [0.8, 1.25] lần scale gốc.
Export vẫn là integer codes/scales và VAE dequantized FP32; không phải fine-tune
trọng số tự do, không phải kernel INT4 thực thi.
LPIPS AlexNet v0.1 đóng băng, nhận RGB [-1,1]; lambda mặc định 0.1.
Gradient perceptual truyền tới quantizer; model-only không có thành phần này.
`--natural-perceptual-weight 0` là ablation natural MSE-only. Trọng số loss bảo
toàn ảnh sinh mặc định 2. Reconstruction-only model-only vẫn là một nhánh riêng,
tái tạo ảnh marked chứ không dùng ảnh natural hay LPIPS.

`natural_qat_purification` mở rộng biến quyết định từ floor/ceil sang offset liên tục
trong đơn vị mã, hard-forward bằng W4 và dùng STE khi backward. Offset bị chặn ở
`--qat-max-code-shift` (mặc định 2), chịu L2 trust penalty, rồi bị làm tròn khi export.
Loss bảo toàn output sinh chỉ so phiên bản low-pass 8x để giữ bố cục/nội dung mà không
ép khớp toàn bộ residual của decoder fingerprint. Owner key/extractor vẫn chỉ xuất hiện
sau khi tất cả checkpoint đã được chọn và đóng băng.

Dataset được deduplicate theo hash file và chia train/search bằng seed cố định;
hash/path và phép resize/crop được ghi manifest. Dedup hash không phát hiện ảnh
gần trùng hoặc ảnh được encode lại: người dùng cần kiểm tra nguồn dataset.
Số ảnh mặc định 32 train + 20 search là pilot, không đủ làm mặc định paper.
Runner chỉ đọc số ảnh đã chọn; cache CPU hoặc GPU theo ngân sách VRAM trống,
không tự nạp toàn bộ dataset ngoài số lượng đã chọn.
Ảnh natural được giả định không watermark theo nguồn dữ liệu, không được owner
detector xác nhận trước training. `prepare_natural_images.py` tự tải metadata COCO
test2017 đã pin checksum và từng ảnh cần dùng qua HTTPS từ bucket COCO, ghi license,
URL và SHA256 ảnh, kiểm tra lại khi reuse. Đây là pool calibration ngoài; tên split
COCO test2017 không biến nó thành tập held-out của thí nghiệm này.

Checkpoint được chọn bằng MSE + lambda LPIPS trên **natural search**.
Mặc định `--quality-policy report`: quality trên **generated search** ghép cặp
chỉ ghi nhận, không lọc candidate hoặc dừng nhánh. `--quality-policy constrained`
là đối chứng ưu tiên quality-feasible; khi mọi candidate trượt vẫn xuất fallback.
Hai tập này khác nhau; không so PSNR giữa ảnh tự nhiên và ảnh sinh không liên quan.
Test chung của các phương pháp vẫn là ảnh sinh từ prompt/seed chưa dùng để chọn.
Mọi nhánh đều được freeze trước khi sinh test và gọi owner evaluator.

### Kiểm chứng ba điểm yếu của model-only

Không thể chứng minh blur loại fingerprint chỉ từ marked model. Thay vì gọi
pseudo-target là sạch, code xuất và đánh giá riêng ba loại ảnh:

| Nhánh ảnh | Câu hỏi |
|---|---|
| `marked_reference_test` | Watermark gốc có được phát hiện tốt không? |
| `pseudo_target_test` | Smoothing trực tiếp có làm giảm detection không? |
| Các nhánh `*_test` của quantizer | Quantizer có thực hiện được mục tiêu mà giữ chất lượng không? |

Owner đo bit accuracy, double-tail TPR/evasion và survival trên cùng từng ảnh.
Mọi loại ảnh dùng cùng quy trình lưu PNG và extractor. `report.json` thêm
high-pass MSE và tỷ lệ năng lượng high-pass (x - Gaussian(x)); CSV owner có
trung bình các metric này, PSNR/SSIM và LPIPS nếu bật. Đây là chỉ báo mất chi tiết,
không phải phép phân tách tần số watermark/texture. `reconstruction` được thêm
vào ladder mặc định làm đối chứng không smoothing, giữ các nhánh trước đây.

- Pseudo-target vẫn có TPR cao: giả thuyết mục tiêu chưa hiệu quả, dù MSE giảm.
- Pseudo-target TPR thấp nhưng output quantizer TPR cao: cần xem khoảng cách
  tới target/khả năng biểu diễn trong grid, chưa thể kết luận quantization thành công.
- TPR giảm cùng chất lượng/texture giảm mạnh: chưa có bằng chứng xóa chọn lọc.
- Sensitivity được ghi rõ `visual_proxy_not_ownership`; không có detector/key
  thì chưa có căn cứ gọi đó là ownership sensitivity. Không dùng owner test để
  xếp hạng layer hoặc chọn lại checkpoint trong cùng thí nghiệm.

Đây là sửa protocol kiểm chứng và thêm mục tiêu độc lập từ dữ liệu tự nhiên,
**không phải chứng minh đã giải quyết triệt để ba giới hạn nhận dạng tín hiệu**.

### Detector hai phía với FPR tổng

Với K bit, số bit khớp M và null iid Bernoulli(0.5), chọn số nguyên nhỏ nhất
k > K/2 sao cho `2 * binom.sf(k-1, K, 0.5) <= FPR`.
Phát hiện nếu `M >= k` **hoặc** `M <= K-k`. Không làm tròn một ngưỡng BA thực
rồi coi FPR là đúng. K=48: FPR tổng 1e-3 cho k=36 (đuôi dưới <=12),
FPR 1e-4 cho k=38 (đuôi dưới <=10). Ngưỡng bất khả thi được báo lỗi;
baseline yếu vẫn xuất báo cáo và được đánh dấu, không hạ FPR để cứu kết quả.
Đây là FPR lý thuyết theo null bit độc lập, chưa phải FPR thực nghiệm.
`DETECTOR=single` chỉ dùng để đối chiếu giao thức cũ, không so ngang TPR hai detector.

### Chạy

```bash
# Cả hai hướng, tự tải dữ liệu và tự owner-evaluate
bash run_blind_quantization.sh

# Chạy cả hai hướng với ảnh natural của bạn
bash run_blind_quantization.sh --natural-images /data/natural_images

# Chỉ model-only nghiêm ngặt
WMQ_MODEL_ONLY=1 bash run_blind_quantization.sh
```

Xem `manifest.json` cho threat model từng nhánh, `search.json` cho mục tiêu chọn,
`report.json` cho quality/texture và `watermark_retention.csv` cho owner metrics.
Ảnh được lưu ở `output_artifacts/images/<run>/`, checkpoint ở
`output_artifacts/checkpoints/<run>/`; model, dataset, extractor và cache cũng nằm
trong `output_artifacts/`. Tất cả tách khỏi `output_attack/<run>/` chỉ chứa báo cáo
và log nhẹ. Manifest ghi các đường dẫn; evaluator vẫn tự đọc và kiểm tra hash.
Phân tích CSV/JSON sau khi hoàn tất không cần kèm ảnh. Run cũ giữ cấu trúc cũ.
`search.csv` giữ các candidate đã đánh giá, `quality_summary.csv` giữ quality
của mọi nhánh, `branches/*/updates.csv` giữ loss từng bước. Quality thấp không
dừng chương trình; lỗi cấu hình/artifact hoặc NaN vẫn được xử lý riêng.
Không chọn hướng/strength tốt nhất từ owner test rồi báo đó là kết quả blind
chưa tuning; nghiên cứu tiếp cần một test pool độc lập.

## Kết luận nghiên cứu

Không có căn cứ từ các nguồn bên dưới để bảo đảm một quantizer không biết key,
extractor hay ảnh không watermark sẽ xóa Stable Signature tốt hơn kết quả cũ.
Vì vậy implementation mới là một phương pháp thử nghiệm có giả thuyết rõ ràng;
hiệu quả phải được đo sau khi khóa cấu hình. Nó mở rộng số biến học được từ vài
control theo group thành quyết định rounding cho từng weight trong VAE decoder.

### Các nguồn đã đối chiếu

- [Stable Signature, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Fernandez_The_Stable_Signature_Rooting_Watermarks_in_Latent_Diffusion_Models_ICCV_2023_paper.html):
  watermark nằm trong latent decoder. Nhánh này do đó tập trung vào VAE; không
  có lý do gắn timestep loss vào VAE decoder vì nó chỉ chạy trên latent cuối.
- [Stable Signature is Unstable, 2024](https://arxiv.org/abs/2405.07145):
  tối ưu decoder với ảnh không watermark và latent ước lượng. Công trình cần
  attacking dataset không watermark, nên không phải baseline tương đương với
  giả định chỉ dùng model và dữ liệu model tự sinh. Đây cũng là fine-tuning,
  không phải chứng minh quantization không detector có hiệu quả.
- [AdaRound, ICML 2020](https://arxiv.org/abs/2004.10568): học rounding thay cho
  làm tròn độc lập từng weight. Implementation dùng hard-forward/STE sigmoid,
  không tái lập nguyên thuật toán AdaRound với stretched sigmoid và annealing.
- [Q-Diffusion, ICCV 2023](https://arxiv.org/abs/2302.04304): calibration theo
  timestep và shortcut split dành cho noise estimation network. Không nên gọi
  nhánh VAE mới là Q-Diffusion hay tuyên bố dùng diffusion trajectory gradients.
- [Rounding-Guided Backdoor Injection, 2025](https://arxiv.org/abs/2510.09647):
  nghiên cứu backdoor qua rounding; không phải bằng chứng trực tiếp cho việc
  xóa watermark Stable Signature dưới threat model không extractor/key.

## Threat model và ranh giới thí nghiệm của hướng model-only

Attacker nhận một Diffusers pipeline đã có fingerprint trong VAE. Attacker có
trọng số/config và được sinh ảnh với prompt/seed tự chọn. Không có checkpoint
không watermark, watermark key, extractor, detection API, ảnh tự nhiên bên ngoài,
hay mạng neural khác trong quá trình lựa chọn quantizer. Biết scheme là Stable
Signature được coi là thông tin công khai. Prompt văn bản không chứa nhãn watermark.

`prepare_marked_fixture.py` thuộc vai trò người tổ chức: tải backbone và checkpoint
VAE đã watermark công khai rồi xuất duy nhất pipeline đã fingerprint. Việc tổ chức
biết backbone không được dùng để so sánh weight/ảnh trong chương trình attacker.
Trong thử nghiệm nghiêm ngặt, chạy attacker trong môi trường chỉ thấy thư mục fixture.

`wmq_blind.py` chỉ load fixture local. `evaluate_blind_watermark.py` là chương trình
chủ sở hữu, chạy sau khi lựa chọn đã kết thúc. Không dùng kết quả owner evaluation
để chọn strength, bitwidth, clip, seed hoặc checkpoint của cùng thí nghiệm blind.
Nếu nghiên cứu tuning với feedback detector, cần khai báo threat model oracle khác.

### Audit luồng dữ liệu

Phiên bản trước đã chỉ dùng train để tính loss và search để gọi `choose`, nhưng
sinh sẵn cả ảnh test. Phiên bản hiện tại chỉ tạo train/search trước optimization;
Sau khi chọn xong mới xuất checkpoint trong `output_artifacts/checkpoints/<run>/branches/`, giữ log
nhánh trong `output_attack/<run>/branches/` và ghi `selection_frozen.json` chứa hash
checkpoint/search, rồi mới sinh latent và ảnh test. Không có bước chọn lại sau
test. `test_used_for_selection=False` là metadata mô tả luồng này, tự nó không
phải bằng chứng.

Kiểm thử CUDA với pipeline nhỏ xác nhận: checkpoint/freeze marker có trước lần
sinh test đầu tiên; `choose` không được gọi sau khi test bắt đầu; thay đổi riêng
prompt và phân phối latent test không đổi lựa chọn hoặc weights xuất. Chương trình
attacker không có input key/extractor và không gọi owner evaluator. Đây là kiểm
chứng implementation đang có, không bảo đảm trước việc sửa source/metadata có chủ ý.

Checkpoint Meta được pin SHA256 từ bytes tải ngày 2026-09-15, không phải chữ ký
do nhà phát hành cung cấp. Backbone revision mặc định vẫn là `main`; pin decoder
không đồng nghĩa toàn bộ backbone đã pin immutable revision. Manifest attacker
ghi hash các file model thực tế để kiểm toán. Evaluator từ chối threshold không
thể đạt (ví dụ key 4 bit và FPR 0.001) thay vì xuất TPR=0 gây hiểu nhầm.

## Phương pháp thử nghiệm

1. Sinh và cache latent cuối từ model đã fingerprint. Chia prompt/seed duy nhất
   thành train, search và test. Giữ UNet/text encoder cố định.
2. Ảnh tham chiếu là `x_w = D_w(z)`; đó vẫn là ảnh có watermark.
3. Tạo pseudo-target `t = (1-rho) * x_w + rho * G(x_w)`, với G là Gaussian 5x5
   cố định và rho mặc định 0.25. Giả thuyết: làm dịu tín hiệu chi tiết có thể
   làm yếu fingerprint trong khi giữ nội dung. Không giả định mọi watermark
   đều là high-frequency; blur có thể chỉ xóa texture và giữ nguyên watermark.
4. Cho mỗi bitwidth/clipping, cố định grid per-output-channel từ marked weights.
   Học một alpha cho từng weight bằng Adam. Forward luôn dùng quyết định hard
   `floor(w/s) + 1[alpha >= 0]`, backward dùng gradient sigmoid (STE). Không
   train full-precision model weights, bias hay norm. Nhánh `rounding_scale`
   học thêm per-channel scale có giới hạn như đã mô tả ở đầu tài liệu.
5. Loss là MSE tới pseudo-target cộng MSE giữa hai ảnh sau Gaussian filtering.
   Cả hai mục tiêu đều không dùng thông tin watermark. Đây là gradient theo
   proxy thị giác, không phải gradient watermark `g_own`.
6. Trên search, mặc định chọn MSE tới pseudo-target nhỏ nhất và báo quality
   theo mean PSNR >=25 dB, mean SSIM >=0.9, min per-image PSNR >=22 dB.
   `--quality-policy constrained` ưu tiên checkpoint đạt các ngưỡng này;
   mặc định `report` không dùng chúng để lọc candidate. Cả hai đều chạy tiếp.
7. Sau khi khóa mọi lựa chọn, sinh test cho marked reference, pseudo-target,
   RTN cùng bits/clip/target coverage và các checkpoint đã chọn. Owner đo
   bit accuracy, double-tail TPR/evasion và LPIPS độc lập. Nếu không có candidate
   hợp lệ, vẫn xuất diagnostic result (`selected_quality_failed` trong report mode,
   `no_feasible_candidate` trong constrained mode), không gọi
   đó là attack thành công. Quality failure vẫn giữ artifact/report để kiểm toán.

Model-only không đồng nghĩa biết được đặc trưng nào trong weights là watermark.
Không có oracle thì một proxy mạnh về thị giác vẫn có thể vô dụng với fingerprint.
Không lấy MSE thấp hoặc bitwidth thấp làm bằng chứng attack thành công.

## So sánh cần chạy

- `--strength 0`: reconstruction-only control, cùng grid/steps/dữ liệu.
- `--strength 0.25`: giả thuyết suppression, chốt trước khi xem owner metrics.
- `fixed_ptq` là RTN control độc lập tại mỗi bits/clip được khai báo;
  checkpoint mọi nhánh đều được khóa trước khi sinh test.
- Nhiều seed và key do người tổ chức chuẩn bị, cùng dataset công khai đã cố định.
- Báo cả candidate không feasible, test quality failure và grid fallback.
- Không so trực tiếp với legacy oracle search rồi kết luận phương pháp yếu/mạnh:
  quyền truy cập, bitwidth, target coverage, dtype và quality gate khác nhau.

## Tính toán và giới hạn

Cache latent giúp không phải chạy lại UNet mỗi candidate: vì chỉ thay VAE decoder,
đầu vào VAE cuối chính là latent từ cùng UNet/prompt/seed. Calibration không cần
backprop qua toàn chuỗi diffusion. VAE và quantizer dùng FP32 để tránh nhầm sai số
FP16 với low-bit PTQ; UNet sinh latent ở FP16. Export là simulated weight PTQ với
weights dequantized FP32, không phải packed INT2/INT4, không có claim tăng tốc kernel.

Mỗi update dùng một ảnh train và học đồng thời mọi rounding variable trong scope.
RAM giữ model và snapshot; references/latent có thể được cache lên GPU theo ngân
sách mặc định 4 GiB, bảo toàn ít nhất nửa tổng VRAM trống khi cấp phát cache.
Inference/evaluation dùng batch tự chọn tối đa 8, giảm khi CUDA OOM, ghi lại trong
runtime report. Batch train vẫn 1, seed từng ảnh và phép chia tập giữ nguyên.
Ghi log được gom theo 10 update, không giảm số candidate hay bước tối ưu.
Batching có thể gây sai số số học nhỏ; cần đối chiếu cấu hình batch 1 khi báo cáo
thí nghiệm nhạy ngưỡng, không khẳng định bitwise identical. VRAM phụ thuộc activations VAE
512x512. Có log peak allocated VRAM và thời gian, nhưng chưa có benchmark RTX 6000.
Rounding full decoder có thể cần nhiều VRAM; `--scope late` là thí nghiệm nhỏ hơn
và phải báo phạm vi khác. Không tự thay scope/precision khi thiếu bộ nhớ.

PSNR/SSIM chỉ là proxy chất lượng; ngưỡng LPIPS cũ không được bảo đảm. Owner có
thể đo LPIPS để kiểm tra kết quả và báo thất bại, nhưng không chọn lại candidate
bằng detector test. Ảnh pseudo-target không bao giờ được gọi là ảnh sạch.
## Bổ sung phân tích sau khi freeze

Giữ nguyên các nhánh quantization, không thêm JPEG/resize hoặc checkpoint resume.
Owner evaluation bổ sung thống kê bit-error ghép cặp và exact-message accuracy.
Phân tích cơ chế độc lập `wmq_owner_mechanism.py` chạy sau freeze và kiểm tra hash
trước/sau: với mỗi layer, đo e = theta_quantized - theta_marked, norm và cosine/
dot với gradient ownership/quality tại quantized endpoint. Quality là RGB MSE
với output marked cùng latent; ownership là squared signed soft agreement, đối
xứng khi đảo toàn bộ bit. Không dùng owner gradient trong attacker selection.

Đây là diagnostic Taylor cục bộ, không phải phép chứng minh ownership subspace
hoặc phân rã nhân quả chính xác. Raw gradient quality tại model gốc bằng 0 nên
không được dùng làm cosine attribution. Mặc định 4 test seeds, điều chỉnh bằng
WMQ_MECHANISM_SAMPLES; JSON/CSV ghi từng sample/layer và giới hạn tái sinh latent.
Kiểm tra FP32/RGB/range/finite tại biên ảnh bảo vệ tính hợp lệ của các metric.
