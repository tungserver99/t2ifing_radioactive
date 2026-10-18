# Đối chứng và protocol mới

## Chạy mặc định

```bash
bash run_blind_quantization.sh
```

Hoặc chạy qua bộ tổng hợp:

```bash
bash run_blind_suite.sh
```

Cả hai mặc định dùng `science`: W4, 4.000 ảnh natural TRAIN, 256 natural SEARCH,
32 prompt TRAIN, 20 SEARCH, 100 TEST, 2.000 update mỗi nhánh học, batch 4,
natural resolution 256. Các seed subspace là 1701/1702/1703; đây là ba phép lấy
subspace, không phải ba lần tái lập độc lập của toàn bộ thí nghiệm.
Mặc định chỉ có một mức `preserve_weight=0.5` cho mỗi seed; mức 2 và 8 của sweep
cũ không tự chạy. Muốn thử mức khác thì truyền `--preservation-weights` tường minh.

| Nhánh | Mục đích |
|---|---|
| `fixed_ptq` W4 | RTN không tối ưu |
| `qk_rotation_ptq` W4 | Đổi cơ sở Q/K giữ nguyên FP32 rồi RTN; model-only, không chọn theo owner |
| `reconstruction` W4 | Tái tạo đầu ra model đã watermark, model-only |
| `natural_rounding` W4 | Natural reconstruction + LPIPS, không residual |
| `natural_residual` W4 | PCA residual |
| `natural_residual_qat_warm` W4 | Khởi tạo từ mã W4 residual đã chọn rồi học code offset; step 0 tái tạo mã nguồn |
| `natural_random_subspace` W4 × 3 | Subspace ngẫu nhiên cùng rank |
| `natural_frequency_subspace` W4 | Các hướng DCT tần số cao, cùng rank |
| `natural_contrastive_subspace` W4 | Hướng residual cao tương đối với texture tự nhiên |
| `natural_full_finetune` FP32 | Đối chứng thay đổi tự do decoder |
| `natural_finetune_rtn` W4 | RTN trực tiếp checkpoint FP32 vừa chọn bằng SEARCH |
| `natural_teacher_rounding` W4 | Học rounding của model gốc theo đầu ra teacher FP32 cùng latent generated |

Có 14 đầu ra phương pháp đã chọn; thêm checkpoint cuối nếu khác checkpoint được
chọn, cùng baseline và pseudo-target diagnostics. Fine-tune → RTN dùng lại quá
trình học FP32, không train lại. Nó giữ bias/norm đã fine-tune và được ghi nhãn
threat model rộng hơn quantizer-only. Khi cộng chi phí phải đọc `shares_training_with`.

Nhánh rounding chỉ đổi hướng làm tròn; nhánh QAT warm dùng code offset nhiều ô
nhưng vẫn xuất W4, không fine-tune bias/norm. Đối chứng residual có cùng rank,
optimizer, lịch learning rate và minibatch; QAT warm thêm 2.000 update sau nguồn
residual nên phải tính cả chi phí nguồn. Chi phí thực tế vẫn khác:
PCA/calibration, gradient diagnostics và FP32 có overhead riêng. W4 là trọng số
lượng tử hóa được giải lượng tử để tính FP32, không phải tuyên bố kernel INT4.

Các phương pháp cũ vẫn có trong `--profile full` hoặc `WMQ_METHOD_SET=full`.
Tất cả profile (`pilot`, `focused`, `science`, `full`) mặc định chỉ lượng tử hóa W4;
W8 vẫn được hỗ trợ khi truyền `--bits 8` tường minh. Teacher/đối chứng FP32 vẫn giữ.
`focused` giữ W4 residual/QAT và đối chứng reconstruction. Để chạy riêng
model-only: `WMQ_MODEL_ONLY=1 bash run_blind_quantization.sh`.

## Teacher → quantizer

Nhánh `natural_full_finetune` chạy trước các natural student khi bật teacher.
Teacher lấy đúng `decoder_fp32.safetensors` của checkpoint được chọn bằng SEARCH,
không lấy checkpoint cuối nếu hai checkpoint khác nhau. Hash được kiểm tra trước
khi cache target; decoder marked gốc được khôi phục sau khi cache, kể cả khi có lỗi.
Không giữ hai VAE cùng lúc trên GPU. Target generated TRAIN/SEARCH được cache một
lần và đưa lên GPU nếu còn đủ VRAM theo giới hạn cache hiện có.

Student khởi tạo lại từ theta_w gốc, chỉ học floor/ceil rounding ở cùng W4,
giữ bias/norm gốc. Objective gồm natural reconstruction + LPIPS, generated
teacher MSE với `--teacher-weight 1`, preservation và quality penalty hiện có.
SEARCH xếp hạng theo objective tương ứng, giữ gate chất lượng so với marked
reference. Không dùng TEST target, key, extractor hay BA để train/chọn teacher
hoặc student. Teacher chưa được mặc định coi là sạch watermark.

Nếu teacher lỗi, student được ghi branch failure; không lặng lẽ thay bằng một
model khác. Nếu teacher không đạt chất lượng nhưng vẫn xuất được theo chính
sách continue, student vẫn chạy và lưu trạng thái `search_feasible` của teacher.
Nếu teacher được chọn ở step 0 thì target có thể trùng model marked gốc.

`selection.json` lưu `teacher_dependency` (label, step, hash, quality, cache cost),
`teacher_search_mse` và số forward thêm. `updates.json/csv` lưu `teacher_mse`.
`suite_results.csv` có teacher source/label/step và MSE. Tổng chi phí phương pháp
là teacher + student; teacher dùng chung với FP32 control nên chỉ cộng một lần.
Đây là protocol cho phép fine-tune model phụ trung gian, nhưng artifact student
chỉ thay quantizer. Không gọi nó là threat model cấm mọi fine-tuning trung gian.

Ablation dùng model marked làm teacher, giữ nguyên loss và budget:

```bash
bash run_blind_suite.sh -- --teacher-source marked
```

Để chạy trực tiếp bộ nhánh nhỏ tập trung vào đối chứng teacher:

```bash
bash run_blind_quantization.sh --methods fixed_ptq reconstruction \
  --natural-methods natural_rounding natural_full_finetune natural_teacher_rounding
```

Nhánh fine-tune → RTN vẫn được tạo mặc định. Dataset vẫn 4.000 ảnh natural,
2.000 step × batch 4 tương đương 8.000 lượt natural TRAIN mỗi nhánh học, chưa
tính forward generated preservation/distillation và validation. Bản thân teacher
distillation không bảo đảm BA giảm; cần đọc kết quả owner cuối run.

## Phần phương pháp mới cần kiểm chứng

Từ ảnh natural TRAIN và VAE đã watermark, tính residual `D(E(x))-x` và patch
texture của `x`, loại DC từng kênh. Nhánh contrastive giải bài toán trị riêng tổng
quát giữa second moment residual và second moment texture có ridge. Sau đó
trực chuẩn hóa các hướng để tạo projector. Không dùng key hoặc extractor.

Đây là giả thuyết giảm nhiễu texture trong mục tiêu, **chưa chứng minh tìm ra
ownership subspace**. VAE reconstruction error cũng xuất hiện trong residual.
PCA, random, DCT và contrastive được chuẩn hóa loss để có cùng năng lượng
projection baseline trên TRAIN; không dùng SEARCH/TEST để tính hệ số. Cờ
`--no-subspace-normalize` cho phép ablation tắt chuẩn hóa. DCT chọn các atom
tần số cao cố định, không phải một control khớp hoàn toàn phổ PCA.

`residual_calibration.json` lưu loại basis, seed, rank, hệ số loss và thống kê.
Thống kê ổn định split-half của PCA không được gán cho random/DCT/contrastive.

## Chất lượng và cách đọc kết quả

Mặc định `--quality-constraint off` và SSIM trung bình tối thiểu `.8` cho mọi nhánh;
PSNR trung bình và PSNR từng ảnh vẫn phải đạt ngưỡng hiện có. Khi bật tường minh
`--quality-constraint dual --budget-psnr 30 --budget-ssim .8`, hệ thống thêm penalty
vi phạm trên từng ảnh generated TRAIN. Hệ số tự tăng khi vi phạm, có giới hạn;
đây là adaptive hinge penalty, không có bảo đảm hội tụ dual hay bảo đảm mọi ảnh
đều đạt ngưỡng. SSIM được tính có gradient. Bộ chọn SEARCH ưu tiên feasible với
không quá `--budget-max-violation .1` số ảnh vi phạm, ngoài các gate vốn có.
Không dùng dual mặc định vì nó sẽ loại điểm `natural_residual` TPR 60/100 ở run
`131250`: trên SEARCH có 8/20 ảnh dưới SSIM .8 và 11/20 ảnh dưới PSNR 30.

**Không dừng toàn bộ run vì chất lượng thấp.** Nếu không có candidate đạt,
vẫn xuất fallback và ghi quality failed. Giữ checkpoint cuối cho phân tích nếu
bật `--evaluate-final`. Lỗi dữ liệu, checkpoint hoặc tính toàn vẹn vẫn phải báo lỗi.

`watermark_retention.csv` thêm `joint_success_rate`: số ảnh vừa né detector vừa
đạt PSNR/SSIM, chia cho **tất cả ảnh reference vốn được phát hiện**. Không bỏ
ảnh xấu khỏi mẫu số. Có khoảng Wilson 95% và `joint_quality_evasion.csv` cho
các ngưỡng SSIM .8/.9/.95 cùng ngưỡng chính. Double-tail vẫn hiệu chỉnh ngưỡng
bit nguyên theo tổng FPR; ngưỡng chất lượng độc lập với ngưỡng detector.

`updates.json` có gradient norm của reconstruction, residual và preservation
mỗi 100 update; hệ số penalty và penalty loss được lưu. Đây là gradient loss
của attacker, không phải owner gradient. Owner mechanism analysis bổ sung tỷ
lệ năng lượng image gradient nằm trong subspace sau khi selection đã freeze.
Các số này là phân tích endpoint, không phải bằng chứng nhân quả. Mặc định
owner mechanism vẫn dùng 4 ảnh; tăng bằng `WMQ_MECHANISM_SAMPLES=16` nếu cần.

Sweep ngân sách chất lượng (thực sự thay loss lúc train, không chỉ đổi nhãn):

```bash
bash run_blind_suite.sh --profile science --seeds 3407 \
  --quality-budgets .8 .9 .95
```

Tái lập development nhiều seed:

```bash
bash run_blind_suite.sh --profile science --seeds 3407 4407 5407
```

Gắn `--plan-only` để chỉ xuất kế hoạch. Báo cáo toàn bộ seed/budget; không chọn
winner trên TEST rồi coi cùng TEST là holdout mới. Một seed hoặc 100 ảnh không
đủ để kết luận ưu thế thống kê chắc chắn.

## Final trên prompt, key và checkpoint mới

Sau khi chốt phương pháp bằng development, tạo prompt chưa trùng với tất cả
manifest development đã dùng:

```bash
python src/prepare_final_prompts.py \
  --development-manifests output_attack/DEV_RUN/manifest.json \
  --output final_prompts.txt
```

Pool là các prompt tự biên soạn tổ hợp, không đại diện mọi phân phối thực tế;
kiểm tra chống trùng dùng văn bản chuẩn hóa, không chứng minh khác biệt ngữ nghĩa.
Khai báo **tất cả** run development liên quan. Chuẩn bị config dựa trên
`src/replication.example.json`, điền đường dẫn Linux thực và key của từng model.
Các checkpoint phải thực sự đã được nhúng key mới; thay chuỗi key trong config
không tạo được model watermark mới. Không có checkpoint/key mới thì chưa thể
tuyên bố cross-key replication. Giữ config chứa key ở local.

```bash
python src/run_blind_replication.py --config replication.local.json \
  --output output_attack/final_plan --plan-only
python src/run_blind_replication.py --config replication.local.json \
  --output output_attack/final_replication
```

Orchestrator từ chối dùng lại key/VAE development, kiểm tra extractor hash,
đóng băng plan và chạy đủ asset × seed. Attack nhận model, prompt và manifest;
key/extractor dành cho owner evaluation cuối run. Source/prompt/VAE bị đổi sau
khi freeze sẽ bị phát hiện trước run tiếp theo. Không dùng owner-informed
steering plan trong final. JSON kết quả ở `output_attack`; ảnh/checkpoint/cache
tiếp tục nằm trong `output_artifacts`. Config mẫu không chứa tài sản chạy thật.

## Kết luận khoa học được phép

Code mới cung cấp các đối chứng cần thiết; nó chưa chứng minh attack mạnh hơn.
Chỉ kết luận residual có giá trị riêng nếu tốt hơn reconstruction/random/DCT
ở cùng chất lượng và ngân sách qua nhiều seed. Nếu fine-tune → RTN tốt hơn,
phải ghi nhận lợi thế của không gian trọng số rộng hơn. Novelty còn cần đối
chiếu literature và kết quả cross-key/checkpoint thực, không chỉ tên phương pháp.
