# Genetic search cho quantizer SleeperMark — 27/09/2026

## Kết quả đang có và động cơ

Suite `sleeper_parallel_20260927_092610_703953` còn chạy khi bắt đầu thay đổi này.
Ở lần kiểm tra đầu, CFG đạt 1.550/2.000 bước, delta equivariance 750,
coherent W4 625 và delta coherent 650. Chưa có owner evaluation của suite đó.
Hiệu chỉnh coherent ở các dòng vừa quan sát chỉ khoảng 1e-7–1e-5 RMS;
không thể suy ra watermark yếu hơn từ việc loss nhỏ.

Run đã hoàn tất `sleepermark_20260927_075826_683759`: conditional rollout
BA 94,23%, TPR 98%, SSIM triggered 0,6323, FID tới marked reference 111,68,
joint success 0/100. Đây chưa phải kết quả tấn công hữu ích khi giữ chất lượng.

## Research và giới hạn novelty

- [NSGA-II, Deb et al., 2002](https://doi.org/10.1109/4235.996017):
  chọn theo non-dominated sorting, giữ elite và dùng crowding để giữ đa dạng.
- [EMQ, ICCV 2023](https://arxiv.org/abs/2307.10554): evolutionary search
  được dùng để tìm proxy cho mixed-precision quantization. Paper cũng cho thấy
  chất lượng proxy là vấn đề quan trọng. Nó không chứng minh proxy watermark
  trong code này hiệu quả.

GA không phải đóng góp mới tự thân. Thử nghiệm này kiểm tra việc tìm trực tiếp
trên lưới W4 rời rạc có lợi hơn random search cùng ngân sách hay không.
Không gọi nó là gradient-steered; không hứa BA về 0,5–0,6. Nếu proxy chỉ đo
texture hoặc tín hiệu thường, GA vẫn có thể tối ưu sai mục tiêu rất tốt.

## Hai nhánh mới (chưa bật vào mặc định)

`genetic_quantizer_w4` và `random_quantizer_w4` cùng dùng:

- Chỉ thay các ma trận UNet được scope chọn, mặc định up-block attention.
  VAE/CLIP và trọng số nguồn giữ nguyên; không fine-tune tự do.
- Mỗi attention module có hai gene nguyên trong [-3,3]. Gene thứ nhất nhân
  scale theo `exp(0.04*g)`; gene thứ hai dịch ngưỡng rounding theo `0.1*g`.
  Mỗi group 64 vẫn có scale nền riêng. Mã nguyên luôn nằm trong [-8,7].
- Chromosome 0 khớp chính xác grouped MSE-init RTN. Forward dùng trọng số
  dequantized FP32, không claim kernel INT4 hay toàn bộ UNet INT4.
- Hai fitness cần giảm: ordinary CFG/unconditional noise reconstruction MSE
  và x0 MSE tới spatial pseudo-target tại late timestep. Target được tính một
  lần từ model marked với prefix ngẫu nhiên độc lập; không dùng owner trigger.
  Phép dịch không gian chỉ dùng tạo target calibration, không hậu xử lý ảnh test.
- FIT và SELECT tách theo prompt trong TRAIN. Mỗi bank mặc định 16 records,
  trải qua các timestep. Tất cả candidate dùng cùng bank và target cố định.
- Population 12, 12 thế hệ con: **156 lần tính fitness** mỗi nhánh. GA có
  tournament selection, uniform crossover, mutation rời rạc, elitist survival
  theo Pareto rank/crowding. Random dùng cùng initial population, miền gene,
  số lần tính fitness và quy tắc chốt. Không có Adam/STE trong search.
- Mỗi chromosome chỉ tính fitness một lần; lai ghép/đột biến tạo chromosome
  đã thử sẽ được thay bằng cá thể mới trước khi gọi UNet. Hai nhánh đều có
  cùng ngân sách chromosome khác nhau. Điều này không bảo đảm mọi chromosome
  tạo trọng số khác nhau (hai cấu hình có thể trùng sau rounding).
- Sau search, kiểm tra chromosome 0 và 11 candidate xếp theo FIT rank/crowding
  trên SELECT. Chọn proxy loss thấp nhất với ordinary MSE không quá 1,25 lần
  RTN trên SELECT. Nếu không có candidate mới tốt, có thể giữ RTN. Đây không
  phải ngưỡng SSIM; bảo toàn ảnh thực phải đo ở owner evaluation.
- Bổ sung kiểm tra từng record SELECT: noise MSE không quá 2 lần RTN của
  chính record đó (`--evolution-tail-ratio 2`). Mẫu có RTN loss gần 0 dùng
  denominator floor được ghi rõ trong JSON. Báo max/p95/violation fraction
  và lưu toàn bộ loss từng record; candidate không đạt vẫn được ghi lại,
  thí nghiệm tiếp tục chạy. Anchor RTN luôn là phương án dự phòng hợp lệ.
- Không dùng TEST, BA, TPR, extractor hay key trong fitness/chọn cấu hình.
  Sau khi freeze, pipeline hiện có tự tính BA, double-tail TPR, chất lượng và
  FID. Không chọn lại chromosome theo kết quả TEST.

## Chạy

```bash
bash run_blind_quantization.sh --watermark sleepermark \
  --methods genetic_quantizer_w4 random_quantizer_w4
```

`--steps` không điều khiển GA. Thay ngân sách bằng `--evolution-population`,
`--evolution-generations`, `--evolution-records`. Ví dụ 12×(12+1) ở trên.
Để so sánh công bằng phải giữ các tham số đó giống nhau giữa GA và random.
Chỉ hai nhánh này chạy song song trong suite mới; xếp suite sau job cũ trên
server, không thay code hay ngắt process đang chạy.

Kết quả nhẹ: `output_attack/sleeper_parallel_*`, gồm `*_training.csv`,
`*_validation.csv`, `*_search.json` (toàn bộ gene/fitness/Pareto, split và
quy tắc chọn), cùng bảng owner/FID. Ảnh và checkpoint vẫn ở các thư mục nặng
riêng theo launcher. Không giữ một bản model GPU cho mỗi cá thể: population
là mảng gene CPU nhỏ, từng cá thể được đánh giá lần lượt trên cùng UNet.

## Cách đọc kết quả

Đầu tiên so GA với random và RTN ở cùng chất lượng, rồi mới so với các QAT
khác. Fitness tốt hơn chỉ chứng minh tìm tốt hơn theo proxy. TPR và joint
success trên test mới trả lời được có lợi cho tấn công hay không. FID với
100 ảnh chỉ là thăm dò; không đủ để khẳng định chất lượng phân phối cho paper.

Hai nhánh có cùng số candidate và forward ngân sách search/selection; thời
gian chạy thực có thể khác do tranh chấp GPU. Không gọi đó là cùng ngân sách
với QAT 2.000 bước. Muốn kết luận về ưu thế phương pháp phải bổ sung nhiều
seed, held-out prompts/keys và đối chứng proxy; không chọn tham số theo TEST.

## Kiểm thử triển khai

14 unit/integration tests đã qua, gồm UNet diffusers thật ở kích thước nhỏ,
khôi phục trọng số sau lỗi, split theo prompt, reproducibility và miền mã W4.
Sau cập nhật log có thêm 5 kiểm thử nhánh evolution chạy lại đều qua.
Smoke test trên H200 với checkpoint UNet SleeperMark chính thức, latent 64×64,
FP32, population 4 và một thế hệ: cả GA/random đều qua, peak allocated khoảng
3,93 GiB cho UNet-only smoke. Đây không phải VRAM toàn pipeline và không phải
kết quả attack: latent/context trong smoke là ngẫu nhiên. Không thay đổi hay
ngắt các tiến trình thí nghiệm đang chạy.
