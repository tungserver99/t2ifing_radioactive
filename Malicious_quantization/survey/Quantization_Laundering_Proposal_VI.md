# Quantization Laundering? Đánh giá xác minh quyền sở hữu diffusion dưới lựa chọn PTQ thích nghi

**Tên tiếng Anh dự kiến:** *Beyond Fixed PTQ: Secret-Agnostic Quantizer Selection for Diffusion Model Ownership Verification*.

**Trạng thái:** proposal nghiên cứu, chưa có kết quả thực nghiệm. Cập nhật ngày 08/09/2026. Chỉ dùng tên khẳng định “Quantization Laundering” cho bài báo nếu thực nghiệm xác nhận hành vi này.

## 1. Luận điểm và phạm vi

Độ bền trước một vài cấu hình post-training quantization (PTQ) cố định chưa chứng minh độ bền trước việc lựa chọn cấu hình có chủ đích trong cùng không gian triển khai. Nghiên cứu kiểm tra liệu một chính sách PTQ được tìm trên các ownership instance do người đánh giá kiểm soát có thể làm thất bại xác minh trên instance có bí mật chưa từng thấy, đồng thời giữ chất lượng sinh ảnh và đáp ứng ràng buộc triển khai.

Đây là **giả thuyết cần kiểm chứng**, không phải kết luận rằng compression tất yếu xóa được watermark. Đối tượng là checkpoint diffusion có thể tải xuống hoặc bản sao checkpoint mà bên xác minh có quyền kiểm tra. Không suy rộng sang mọi API sinh ảnh hoặc mọi cơ chế provenance.

Kết quả mong muốn có ba phần: tìm được trường hợp thất bại; đo mức phổ biến dưới một phân phối cấu hình công bố trước; và đo khả năng tìm được thất bại trong ngân sách hữu hạn mà không dùng victim verifier.

## 2. Cơ sở và khoảng trống nghiên cứu

| Nghiên cứu/nhóm phương pháp | Điều đã có | Hệ quả cho proposal |
|---|---|---|
| SoK về DNN watermarking, IEEE S&P 2022 | Toolbox chính thức đã có weight quantization trong model-modification attacks | Không nhận quantization-as-attack là ý tưởng lần đầu |
| Cert-LAS, 2026 | Appendix E thử TorchAO W8A32/W8A8/W4A32; báo VSR còn tốt ở compression vừa phải | Phải tái lập baseline và xét adaptive PTQ; không mặc định phương pháp này yếu |
| Collapsed Generation, arXiv 08/2026 | Thử BF16/INT8/FP4; Appendix C, Table VIII có so sánh FingerInv | Cần đối chiếu cấu hình, giao diện verifier và tiêu chí chất lượng, không chỉ thêm hàng INT4 |
| Q-Diffusion/PTQD | PTQ dành cho diffusion đã xử lý đặc tính quá trình denoising | Lựa chọn calibration và precision phải dựa vào khả năng backend thật |
| SVDQuant | Dùng nhánh low-rank độ chính xác cao kết hợp nhánh low-bit | Là một hệ triển khai mở rộng; không nằm trong tập chính nếu cấm thêm nhánh |

Nguồn: [SoK toolbox](https://github.com/dnn-security/Watermark-Robustness-Toolbox), [Cert-LAS, Appendix E](https://arxiv.org/html/2605.29809v1#A5), [Collapsed Generation, PDF và phụ lục](https://arxiv.org/pdf/2608.11732), [Q-Diffusion](https://github.com/Xiuyu-Li/q-diffusion), [PTQD](https://github.com/ziplab/PTQD), [SVDQuant](https://arxiv.org/abs/2411.05007).

**Khoảng trống đề xuất:** đánh giá và tìm kiếm PTQ thích nghi, có ràng buộc utility, trong không gian triển khai hữu hạn và có kiểm chứng backend, rồi đo transfer sang bí mật sở hữu chưa từng thấy. Việc rà soát hiện tại chưa đủ để tuyên bố “first”; cần cập nhật related work trước khi nộp bài.

**Đóng góp thuật toán dự kiến:** dùng độ nhạy sở hữu–utility trên surrogate để phân bổ ngân sách tìm kiếm. Beam search tự nó không mới; giá trị phải được chứng minh bằng hiệu quả hơn random search và các ablation cùng ngân sách, hoặc bằng phát hiện security có sức nặng độc lập.

## 3. Câu hỏi nghiên cứu và giả thuyết có thể bác bỏ

- **RQ1 — Adaptive gap:** trong cùng resource envelope và cùng tập cấu hình hợp lệ, adaptive PTQ có làm xác minh suy giảm hơn fixed PTQ, utility-only search và random search không?
- **RQ2 — Secret-agnostic transfer:** cấu hình được chọn mà không dùng bí mật/verifier của victim có chuyển giao sang các ownership instance chưa thấy không? Tách cross-message, cross-extractor, cross-fingerprint và cross-model.
- **RQ3 — Mechanism:** block/component nào làm verification nhạy hơn utility? Tín hiệu nằm trong weights bị làm tròn hay phụ thuộc activation/runtime? Profiling có dự đoán được tác động trên dữ liệu độc lập không?
- **RQ4 — Recoverability:** khi bên xác minh đổi sang runtime độ chính xác cao trên cùng weights đã giải lượng tử, hoặc tăng ngân sách xác minh theo protocol định trước, kết quả thất bại có còn không?

Không tìm thấy cấu hình thành công trong ngân sách B chỉ là kết quả thực nghiệm hữu hạn. Nó không chứng minh không tồn tại tấn công.

## 4. Đối tượng và giao diện xác minh

Gọi một instance sở hữu là \(i=(m,\theta_i,s_i,\mathsf{Ver}_i)\), với phương pháp \(m\), checkpoint \(\theta_i\), thông tin bí mật \(s_i\) và thuật toán xác minh. “Ownership instance” không đồng nghĩa với một khóa mã hóa độc lập.

| Phương pháp | Carrier/giao diện | Đơn vị bí mật hoặc instance phải giữ riêng |
|---|---|---|
| AquaLoRA | Thông tin được tích hợp vào U-Net qua watermark LoRA, đọc từ ảnh sinh ra | Message, cấu hình watermark/extractor, checkpoint đã nhúng |
| FingerInv | Fingerprint không xâm lấn; latent/trajectory dùng trong denoising để tái tạo ảnh tham chiếu | Ảnh tham chiếu và fingerprint tương ứng; nhiều fingerprint có thể dùng chung weights |
| Cert-LAS | Xác minh theo cơ chế classifier và smoothing riêng của phương pháp | Classifier/private configuration, dữ liệu xác minh, checkpoint watermark |
| Stable Signature, mở rộng | Carrier nằm ở decoder | Message, decoder đã nhúng và extractor |

Đặc tính nguồn: [AquaLoRA chính thức](https://github.com/Georgefwt/AquaLoRA), [FingerInv, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Teng_Fingerprinting_Denoising_Diffusion_Probabilistic_Models_CVPR_2025_paper.pdf), [Cert-LAS](https://arxiv.org/abs/2605.29809), [Stable Signature](https://github.com/facebookresearch/stable_signature).

Với FingerInv, quyền gọi denoiser trong một pipeline điều khiển được khác với quyền gọi text-to-image API chỉ trả ảnh cuối. Ghi rõ control/observation thật, không chỉ dùng nhãn “black-box”. Tất cả thí nghiệm chính giữ nguyên sampler, số bước, CFG và resolution đã chọn cho từng phương pháp; không tính lỗi do bỏ giao diện xác minh bắt buộc là laundering.

Với AquaLoRA, chuẩn hóa checkpoint sau merge LoRA; không tính thao tác gỡ adapter là PTQ attack. Với Stable Signature, quantize U-Net mà giữ decoder nguyên vẹn là control theo carrier, không phải thử nghiệm trực tiếp đầy đủ lên carrier của nó.

## 5. Threat model và security game

### 5.1 Main setting: có weights, không có bí mật sở hữu

Attacker có checkpoint victim và biết mô tả công khai của phương pháp sở hữu. Attacker được tạo surrogate instance, dùng dữ liệu calibration công khai và chọn cấu hình trong registry PTQ. Không biết \(s_v\), không query victim verifier, không nhận victim scores trong profiling, pruning, search hoặc chọn checkpoint cuối.

Tách hai mức truy cập:

1. **Checkpoint-access, secret-agnostic:** biết weights victim nhưng thiếu bí mật/verifier riêng. Không gọi đây là weight black-box.
2. **Oracle-assisted reference:** được biết bí mật victim hoặc nhận score verifier trong search. Score access chưa có nghĩa là truy cập gradient/white-box hoàn toàn. Đây là reference mạnh hơn về thông tin, không phải cận trên được chứng minh của thuật toán heuristic.

Việc viết \(\min_Q V(Q)\) chỉ mô tả mục tiêu, không tự động cấp quyền query \(V\). Quyền truy cập phải được nêu riêng như trên.

### 5.2 Freeze và áp dụng

Tìm configuration/policy \(c^*\) trên surrogate, khóa manifest và hash trước khi mở victim results. Khi áp dụng lên victim, backend được phép tính lại scale/statistics từ weights hoặc public calibration bằng **quy tắc đã freeze**. Không bắt buộc copy scale số học của surrogate; không được đổi policy dựa trên victim ownership score.

Trong protocol chính, chọn đúng một configuration trước khi kiểm tra victim. Nếu utility trên victim không đạt, ghi thất bại joint success. Một nhánh mở rộng có thể cho phép fallback chỉ dựa trên victim utility, nhưng thứ tự và quy tắc phải freeze trước, báo riêng.

### 5.3 Phân biệt nguồn surrogate

- **Watermark:** thay message với cùng extractor chỉ là cross-message. Muốn nói transfer sang cơ chế bí mật độc lập phải giữ riêng extractor/classifier hay các thành phần bí mật thực sự của phương pháp.
- **Fingerprint:** tạo các fingerprint riêng trên checkpoint victim trước PTQ có thể là hành vi hợp lệ vì attacker có weights. Báo là *same-checkpoint, unseen-fingerprint transfer*; không gọi là cross-model.
- **Cross-model:** search hoàn toàn trên các source checkpoint khác rồi áp policy lên victim; layer mapping phải được khóa trước. Bắt đầu cùng kiến trúc. Khác kiến trúc là mở rộng riêng.

### 5.4 Biến đổi được phép

PTQ calibration/reconstruction vốn có của backend được phép với objective utility chuẩn, lịch và ngân sách cố định. Không cập nhật weights tự do bằng ownership loss; không fine-tune victim; không QAT; không sửa trực tiếp rounding decision của từng weight để né verifier; không đổi kiến trúc chức năng, gỡ carrier, thay decoder hoặc sửa ảnh sau sinh trong main setting.

Việc tạo watermark surrogate/owner có thể cần training theo phương pháp gốc. Lệnh cấm training áp dụng cho bước biến đổi victim, không ngăn xây dựng đối tượng thí nghiệm. Chi phí tạo surrogate vẫn phải báo cáo.

### 5.5 Quyền của bên xác minh

Bên xác minh dùng secret/reference nguyên bản và thuật toán đúng của từng phương pháp. Nếu có checkpoint, họ được giải lượng tử weights rồi chạy đường xác minh FP16/FP32 đã đăng ký trước. Kiểm tra cả runtime lượng tử hóa và đường kiểm tra này:

- Chỉ runtime lượng tử hóa thất bại: **runtime-dependent verification evasion**.
- Weights đã giải lượng tử vẫn thất bại: bằng chứng mạnh hơn về suy giảm tín hiệu do PTQ weights.

Giải lượng tử không phục hồi các bit weights đã mất; nó loại bỏ ảnh hưởng riêng của activation quantization và kernel/runtime. Với Cert-LAS, kiểm tra điều kiện/phạm vi certificate; thất bại ngoài miền chứng nhận không bác bỏ định lý của phương pháp.

## 6. Không gian PTQ triển khai hợp lệ

Vì activation quantization biến đổi cả execution, dùng ký hiệu:

\[
M_{i,c}=\mathsf{Deploy}_{h,b}(\theta_i;c,D_{cal}),
\]

với hardware \(h\), backend/version \(b\), configuration \(c\). Ký hiệu \(Q(\theta)\) chỉ là dạng rút gọn, đặc biệt thích hợp với weight-only PTQ.

\[
\mathcal C_{deploy}^{h,b}=\{c\in\mathcal C_{registry}:\operatorname{Valid}_{h,b}(c)=1\}.
\]

Registry chỉ nhận entry có schema hữu hạn, export/reload chạy được và kiểm chứng đúng numerics. Không lấy tích Descartes của mọi option rồi mặc định đều deploy được.

| Trường | Phương án khởi đầu; phải lọc theo backend |
|---|---|
| Scope | Danh sách block/operator cụ thể của U-Net; decoder là track riêng |
| Weight format | INT8; INT4 khi operator/hardware hỗ trợ; FP16 cho block được giữ lại |
| Activation | FP16 hoặc FP32 ghi đúng thực tế; INT8 chỉ ở track hỗ trợ |
| Granularity | Per-channel hoặc per-group; group size 32/64/128 nếu hỗ trợ |
| Clipping | Mặc định/min-max; percentile 99.9/99.99 chỉ khi backend triển khai đúng |
| Calibration | Public prompt IDs; 128 hoặc 256 prompts; seed list cố định |
| Timestep sampling | Uniform hoặc lịch có sẵn của phương pháp; không chỉnh tùy ý theo victim secret |
| Reconstruction | RTN hoặc routine reconstruction chuẩn của backend, budget cố định |
| Runtime | Phiên bản thư viện, kernel, device, accumulation dtype, sampler |

Đây là schema dự kiến, không phải khẳng định TorchAO hỗ trợ mọi tổ hợp trên SD1.5. TorchAO có nhiều tầng algorithm/tensor/kernel; khả năng chạy phụ thuộc dtype/operator/hardware. [Tài liệu chính thức](https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html).

Chia bằng chứng thành hai tầng: **numerical PTQ** (fake-quant/dequantized arithmetic phục vụ phân tích) và **deployment-validated PTQ** (artifact đã export/reload, kiểm tra dtype/coverage và đo hệ thống). Chỉ tầng sau hỗ trợ claim deployment thực tế. Không suy speedup từ bit-width.

Khóa resource envelope \(r=(M_{max},T_{max})\): peak VRAM và latency tối đa trên cùng GPU, batch, resolution. Báo thêm serialized size, precision coverage và overhead. FP16 là control; các phương pháp tìm kiếm so sánh trong cùng envelope. Chọn giá trị cụ thể sau phép đo benign trên hardware thật và trước attack search; hiện chưa biết GPU nên không điền số giả.

## 7. Định nghĩa utility và thành công

### 7.1 Utility so với checkpoint được bảo vệ trước PTQ

Gọi \(M_{i,0}\) là baseline của cùng ownership instance. Với prompt/seed ghép cặp, định nghĩa các độ suy giảm có chiều “lớn hơn là xấu hơn”:

\[
d_F=\mathrm{FID}(M_{i,c})-\mathrm{FID}(M_{i,0}),\quad
d_C=\mathrm{CLIP}(M_{i,0})-\mathrm{CLIP}(M_{i,c}),
\]

\[
d_P=\mathbb E_{p,z}[\mathrm{DreamSim}(M_{i,c}(p,z),M_{i,0}(p,z))].
\]

\[
\operatorname{PassU}_i(c)=
\mathbf1[d_F\le\epsilon_F\land d_C\le\epsilon_C\land d_P\le\epsilon_P].
\]

FID dùng cùng real reference và sample count; không gọi khoảng cách giữa hai bộ ảnh sinh là FID chất lượng với dữ liệu thật. DreamSim/LPIPS là độ tương đồng ảnh ghép cặp, không tự nó chứng minh chất lượng hoặc diversity. Báo KID và diversity theo prompt để phát hiện collapse; paired metric bổ sung cho FID/CLIP.

**Cách khóa ngưỡng:** trước attack, đo repeatability baseline và các benign PTQ trên development prompts, chọn tolerance có ý nghĩa ứng dụng và kiểm tra bằng đánh giá ảnh mù. Điền số vào protocol rồi hash. Chưa biết bộ dữ liệu, metric implementation và GPU nên các \(\epsilon\) hiện là tham số phải hoàn tất ở pilot A; tuyệt đối không chọn lại sau khi xem attack results. Dùng một vector chính và tối đa hai mức phụ đã định trước.

Để kết luận giữ utility trên test, dùng upper confidence bound một phía của độ suy giảm; các constraint đồng thời dùng correction đã đăng ký. Ở pilot, CLIP/DreamSim ít mẫu chỉ là screening, không thay thế final utility audit.

### 7.2 Verification và joint success

\(\mathsf{Ver}_{s_i}(M;\xi,n_v)\in\{0,1\}\) là một quyết định ở **cấp model/trial**, với randomness \(\xi\) và ngân sách xác minh \(n_v\) đã khóa. Đặt:

\[
p_i(c)=\Pr_\xi[\mathsf{Ver}_{s_i}(M_{i,c};\xi,n_v)=1].
\]

Chọn trước \(\beta_{clean}\) và \(\beta_{attack}\), ví dụ mục tiêu pilot là 0.95 và 0.20. Chỉ instance có clean verification đủ mạnh mới thuộc tập \(\mathcal I_0\); vẫn báo số và lý do loại. Thành công ổn định:

\[
L_i(c)=\mathbf1[\operatorname{PassU}_i(c)=1\land
p_i(c)\le\beta_{attack}],\qquad i\in\mathcal I_0.
\]

\[
\mathrm{JLSR}=\frac{1}{|\mathcal I_0|}\sum_{i\in\mathcal I_0}L_i(c_i^*).
\]

Các xác suất được ước lượng bằng repeated trials kèm interval, không lấy một verification failure ngẫu nhiên làm bằng chứng đã xóa ownership. Với verifier quyết định cố định, quy tắc rút về pass/fail. Báo utility pass rate và verification degradation riêng để giải thích JLSR, nhưng không loại các attack làm hỏng utility khỏi mẫu số JLSR.

### 7.3 Threshold và false positives

Khóa thuật toán, ngưỡng, reference, smoothing và ngân sách verifier trước search. Native protocol của từng phương pháp là kết quả chính; một track FPR chuẩn hóa dùng quyết định ở cùng cấp model/trial và phải được ghi là bổ sung, tránh tùy tiện thay thế verifier gốc.

Pilot chuẩn hóa có thể đặt FPR mục tiêu 1%; mức thấp hơn cần đủ null trials hoặc một kiểm định được biện minh. Calibration null và audit null phải tách riêng. Null gồm unwatermarked models, wrong ownership instances và các model cùng họ khi phù hợp với claim cần phân biệt. Không coi các image của một model là các owner/model độc lập.

FPR phải audit lại trên cả clean null và quantized null với ngưỡng đã khóa. Không suy empirical FPR \(10^{-6}\) từ vài nghìn mẫu; nếu dùng mức ý nghĩa lý thuyết phải nêu giả định kiểm định. Với zero errors trên \(n\) Bernoulli trials độc lập, cận trên một phía 95% xấp xỉ \(3/n\); sự phụ thuộc làm diễn giải này không còn tự động đúng.

## 8. Vulnerability region và searchability

Với instance sạch đủ điều kiện và resource envelope \(r\):

\[
\mathcal A_{\epsilon,r}^{(i)}=\{c\in\mathcal C_{deploy}^{h,b}:\operatorname{PassU}_i(c)=1,
\operatorname{Cost}_{deploy}(c)\le r\},
\]

\[
\mathcal V_{\epsilon,r}^{(i)}=\{c\in\mathcal A_{\epsilon,r}^{(i)}:p_i(c)\le\beta_{attack}\}.
\]

Nếu registry hữu hạn đã canonicalize và enumerate hết, dùng tỷ lệ đếm. Khi lấy mẫu, định nghĩa rõ phân phối \(\pi\):

\[
VR_{\epsilon,r}^{(i)}(\pi)=
\Pr_{c\sim\pi}[c\in\mathcal V_{\epsilon,r}^{(i)}\mid c\in\mathcal A_{\epsilon,r}^{(i)}].
\]

Pilot lấy mẫu uniform trên **danh sách config ID hợp lệ đã loại trùng**. Các alias trùng một artifact không được làm tăng trọng số. Đây là prevalence trong registry nghiên cứu, không phải tần suất ngoài thị trường. Báo cả \(\Pr_\pi[\mathcal A]\), số admissible, interval và độ nhạy với một phân phối khác đã đăng ký. Nếu không có admissible config, VR là không xác định, không gán bằng 0.

Lấy mẫu để ước lượng VR độc lập với adaptive search; không dùng tỷ lệ thành công trong candidates đã được thuật toán ưu tiên làm ước lượng volume toàn không gian. Nếu chỉ có một subset thì nêu đúng subset đó.

Đối với thuật toán \(\mathcal S\):

\[
P_{succ}(B)=\Pr_{i,\omega}[L_i(\mathcal S_B(\mathcal I_s;\omega))=1],
\]

trong đó \(\omega\) là randomness search và \(i\) là victim chưa thấy; main setting không dùng victim score để chọn output. Báo trên các prefix budget định trước, ví dụ 32/64/128/256 candidate evaluations. Xác suất “tồn tại candidate tốt trong các cấu hình đã thử” chỉ có thể xác định hậu nghiệm bằng oracle; không thay nó cho thành công của policy keyless thực sự trả về.

## 9. Phương pháp tìm kiếm cụ thể

### 9.1 Score surrogate và profiling

Mỗi method có score \(v_i(c)\) được đổi chiều sao cho lớn hơn nghĩa là evidence mạnh hơn. Chuẩn hóa bằng calibration statistics riêng của surrogate:

\[
a_i(c)=\frac{v_i(c)-\tau_i}{s_i^{score}+\eta},\quad
J(c)=\frac{1}{|\mathcal I_s|}\sum_i a_i(c),
\]

với scale cố định \(s_i^{score}>0\) từ development data; không dùng victim statistics. Tối ưu riêng từng method trong main experiments. Raw bit accuracy, QR reconstruction distance và Cert-LAS score không được cộng trực tiếp với nhau.

Chọn một config benign \(c_0\) theo utility/resource, rồi thay precision/recipe một block mỗi lần. Đặt \(g_{l,q}\) là mean giảm standardized verification margin so với \(c_0\). Dùng proxy suy giảm utility đã chuẩn hóa, không âm \(u_{l,q}\), rồi:

\[
R_{l,q}=\frac{\max(0,g_{l,q})}{u_{l,q}+\delta}.
\]

Ước lượng bằng paired prompts/seeds. Chỉ giữ block có effect lớn hơn mức nhiễu đã đo; chặn denominator và kiểm tra ranking trên validation. Tỷ số thô có thể nổ khi utility gần zero hoặc đổi dấu; đây là heuristic screening, không phải định luật nhân quả.

### 9.2 Pruning và beam search có budget

Pilot mặc định: tối đa 16 block groups cố định theo kiến trúc, top-k = 4, beam width = 4. Nếu model có ít nhóm hơn thì dùng toàn bộ. Khóa seed list và tie-break theo config ID. Search chỉ dùng những trạng thái có trong registry.

```text
Input: surrogate-search/validation instances, valid registry C,
       resource envelope r, utility thresholds, total budget B
1. Chọn c0 từ benign utility-only calibration; đánh giá và charge cost.
2. Profiling tối đa floor(B/4) candidate configs trên surrogate-search.
3. Xếp hạng block theo R; chọn top-k và loại recipe không hợp lệ.
4. Beam = {c0}. Cache mọi configuration đã đánh giá.
5. Dành tối đa floor(B/8) evaluations cho bước validation cuối.
6. Trong budget search còn lại:
   a. Sinh hàng xóm bằng một thay đổi block/recipe hợp lệ.
   b. Bỏ config trùng; đánh giá utility proxy và surrogate score.
   c. Dành 20% lượt cho exploration từ registry ngoài top-k.
   d. Giữ beam width config theo feasibility rồi J(c), tie-break cố định.
7. Full check trên surrogate-validation cho shortlist trong budget dự phòng.
8. Chọn một c* admissible có J validation nhỏ nhất; nếu không có,
   trả failure hoặc c0 với nhãn no feasible attack, theo rule đã khóa.
9. Freeze c*, export/hash manifest; áp dụng một lần lên mỗi held-out victim.
10. Evaluator chạy utility và verifier victim, không phản hồi về search.
```

Feasibility dùng surrogate utility proxy trong search; shortlisted configs phải vượt full utility check trên validation. Proxy không được dùng để chứng nhận final utility. Exploration giảm nguy cơ bỏ lỡ tương tác block; vẫn không bảo đảm tìm optimum toàn cục.

Một candidate evaluation là config được chấm trên một bundle surrogate/data cố định. Candidate count không đủ phản ánh chi phí: log số model evaluations, ảnh sinh, denoiser forwards, verifier calls, GPU-hours và calibration time. Toàn bộ profiling và validation tính vào B; không cấp profiling miễn phí cho phương pháp đề xuất. Random/utility-only baselines dùng cùng fidelity schedule và cùng budget giới hạn; báo thêm kết quả matched GPU-hours.

## 10. Thiết kế thực nghiệm tối thiểu

### 10.1 Ưu tiên

Pilot trên một checkpoint family U-Net tương thích với code ownership gốc; SD1.5 là ứng viên, không ép port nếu làm thay đổi verifier. Bắt đầu AquaLoRA và FingerInv; Cert-LAS là strong target sau khi tái lập clean protocol. Collapsed Generation là đối chứng gần cần đưa vào nếu code/reproduction khả thi. Chỉ mở SDXL hoặc transformer sau khi có tín hiệu pilot.

Code công khai không đồng nghĩa đã sẵn checkpoint và chạy ngay. Gate A phải kiểm tra artifact, dependency và chi phí sinh ownership instances. [Cert-LAS repository](https://github.com/Leyi-Qi/Cert-LAS) hiện mô tả training, VSR và radius evaluation; cần kiểm tra đường thực thi dùng cho compression trước khi đặt lịch chạy.

### 10.2 Splits và ngân sách pilot

| Thành phần | Thiết kế khởi đầu |
|---|---|
| Public calibration | 128–256 prompts, tách utility test và ownership challenges |
| Utility search | 128 prompts × 2 seeds; chỉ screening |
| Utility validation | 512 prompts × 2 seeds, disjoint prompts |
| Final utility audit | 5,000 prompts × 2 seeds cho candidate cuối và baseline; reference dataset khóa trước |
| Surrogate instances | 4 search + 2 validation nếu chi phí cho phép |
| Held-out instances | 10 để thăm dò; chưa đủ kết luận về generalization rộng |
| Search budget | B = 256; báo các prefix 32/64/128/256 |
| Search repetitions | 3 seeds pilot; ít nhất 5 nếu mở full study |
| Verification | Native n_v và số repeated trials khóa ở Gate A |

Đây là cấu hình đề nghị để ước lượng chi phí, không phải claim đã chạy. Với watermark có training đắt, bắt đầu ít independent instances để debug; không lấy nhiều message cùng extractor giả làm nhiều extractor độc lập. Full study dùng power/interval analysis từ pilot để chọn số independent owner/model instances. Không tăng số ảnh rồi coi như đã tăng số model.

Ước lượng GPU-hours từ microbenchmark calibration, generation và native verification; bao gồm tạo surrogate. Cache chỉ những phần không gây rò rỉ hoặc thay đổi PTQ semantics. Không cam kết số ngày/GPU khi chưa đo thực tế.

### 10.3 Baselines bắt buộc

1. Unquantized native precision và benign fixed PTQ recipes.
2. RTN/TorchAO triển khai được trên operator set đã chọn.
3. Q-Diffusion và một diffusion PTQ thứ hai có đường chạy phù hợp, ưu tiên PTQD sau compatibility audit; không bắt buộc thêm tên phương pháp chưa kiểm tra.
4. Uniform random search trên cùng registry, cùng ngân sách.
5. Utility-only selection trong cùng resource envelope; evaluator kiểm tra ownership sau khi freeze.
6. Ownership-aware search không profiling và random-block profiling để đo giá trị ranking.
7. Oracle-assisted adaptive search, báo riêng thông tin được cấp.

GPTQ/AWQ chỉ thêm khi backend hỗ trợ workload thật và có lý do so sánh. SVDQuant là extended deployment track có low-rank branch được phép, cùng resource accounting, không trộn với main threat model.

### 10.4 Ablations và kiểm soát nguyên nhân

- Cùng bit allocation, đổi clipping/calibration; cùng resource envelope, đổi block allocation.
- Profiling vs no profiling; top-k vs random blocks; single-surrogate vs ensemble.
- Weight-only vs activation-only nếu backend cho phép vs weight+activation.
- Runtime lượng tử hóa vs cùng weights dequantized chạy FP16/FP32.
- U-Net vs decoder; với block bị nghi ngờ, khôi phục precision rồi đo xem verification có phục hồi không. Chỉ dùng victim secret cho phân tích hậu nghiệm, không đưa kết quả trở lại main search.
- Plot margin/utility theo timestep dùng dữ liệu độc lập. Tương quan layer sensitivity chỉ là mô tả cho đến khi có can thiệp khôi phục/block-swap.
- Verifier budget chính và một mức tăng đã đăng ký. Ownership failure biến mất khi tăng budget cần được báo như giới hạn của attack.

### 10.5 Báo cáo và uncertainty

Đơn vị độc lập chính là owner/model instance. Bootstrap theo cụm instance; prompts/seeds nằm bên trong, không giả độc lập tất cả ảnh. Với cùng checkpoint và nhiều fingerprint, nêu giới hạn suy rộng. Báo paired interval cho chênh lệch adaptive–random, clean detectability, null FPR, JLSR, utility pass rate, runtime/resource và search cost.

Các bảng phải chứa cả config fail utility, fail backend và search không tìm được nghiệm, với denominator rõ. Không chỉ trình bày vài ảnh đẹp. Một bảng main gồm method, interface, B, JLSR, utility metrics, FPR audit, memory, latency; một bảng transfer tách các loại bí mật và checkpoint.

## 11. Pilot và quyết định tiếp tục/chuyển hướng

| Gate | Việc phải hoàn tất | Quyết định |
|---|---|---|
| A — Reproduction | Clean ownership đủ mạnh; native verifier đúng; ít nhất một benign PTQ hợp lệ; khóa epsilon/FPR/budget/hardware/registry | Chưa đạt thì sửa reproduction; chưa đánh giá ý tưởng attack |
| B — Existence | Fixed/random/oracle search trong registry nhỏ; utility và dequantized verification đầy đủ | Chỉ fail do chất lượng hoặc sai giao diện thì chưa có laundering |
| C — Transfer | Freeze trên surrogate; held-out secret audit; adaptive vs random cùng B và resource | Có transfer lặp lại và lợi ích có ý nghĩa thì mở full study |
| D — Generality | Thêm independent ownership instances và một model family/strong target nếu khả thi | Chỉ chốt claim theo đúng phạm vi được dữ liệu hỗ trợ |

Một mốc thực dụng **để quyết định đầu tư**, không phải chuẩn chấp nhận bài: clean verification ≥ 95%; attack verification ≤ 20% trên các instance thành công; utility pass; có ít nhất vài held-out instances thành công lặp lại; adaptive cải thiện JLSR tối thiểu 20 điểm phần trăm so với random trong pilot. Full study cần interval và effect size, không chỉ mốc point estimate. Các mốc phải khóa trước pilot attack.

Nếu chỉ oracle thành công, hạ claim thành oracle-assisted vulnerability analysis và kiểm tra tính thực tế của quyền truy cập. Nếu random cũng mạnh, attack vẫn đáng quan tâm nhưng không nhận đóng góp search efficiency chưa được chứng minh. Nếu nhiều method vững trong ngân sách đã thử, báo negative finding hữu hạn; không diễn giải thành certificate.

## 12. Hướng thay thế có điều kiện

**Deployment-Induced Verification Drift and Recovery**: nếu PTQ làm dịch chuyển score nhưng “laundering” không ổn định, chuyển câu hỏi sang khả năng duy trì kiểm định sở hữu đúng khi runtime/precision thay đổi.

Nghiên cứu này đo cả false negatives và false positives dưới deployment shift; so sánh verifier native với fixed canonical dequantized execution và một quy tắc hiệu chỉnh dùng owner-side calibration models, không dùng attack test results. Kiểm tra recovery ở FPR giữ cố định và chi phí xác minh hợp lý.

Đây cũng chỉ là hướng dự phòng, chưa được xác nhận novelty. Threshold recalibration đơn giản thường chưa đủ tạo contribution lớn; cần failure mode có tính hệ thống, phương pháp recovery cụ thể, đối chứng cùng FPR và bằng chứng trên nhiều instance. Nếu không có shift đáng kể lẫn attack khả thi, nên dừng đầu tư hướng này thay vì ép thành một defense paper.

## 13. Đóng góp dự kiến và tiêu chuẩn viết bài

Chỉ sau khi có dữ liệu mới chuyển các mục sau thành đóng góp đã đạt:

1. Protocol đánh giá PTQ-adaptive ownership verification với threat model và giao diện rõ ràng.
2. Phát hiện có thể tái lập về secret-agnostic failure dưới utility/resource constraints, nếu tồn tại.
3. Đo prevalence theo registry distribution và budgeted searchability; định lượng lợi ích của sensitivity guidance.
4. Giải thích carrier/runtime và kiểm tra recovery giúp xác định giới hạn thực tế của failure.

Đủ dữ liệu để bắt đầu viết một paper đầy đủ khi có reproduction tin cậy, held-out result, baselines cùng budget, utility/FPR audit và phân tích cơ chế. Bản proposal tốt không thay thế các bằng chứng này. Security venues phù hợp nếu trọng tâm là khả năng né xác minh thực tế; CV/ML venues phù hợp hơn nếu có đóng góp phương pháp và generalization. Chưa có cơ sở hứa hẹn venue cụ thể.

## 14. Artifact và thông tin cần khóa trước experiment

- Checkpoint hashes; code commits; environment; hardware; operator coverage.
- Config registry hữu hạn, canonical config IDs, sampling distribution và resource envelope.
- Tất cả split IDs: calibration, utility search/validation/test, ownership search/validation/test, null calibration/audit.
- Native verifier settings; standardized-track alpha/tau; n_v; repeated trials; utility epsilon và CI rule.
- Search manifest, budgets, seed lists, candidate logs và timestamp/hash freeze.
- Raw utility/verification/null results; export/reload evidence; deployment measurements.
- Liệt kê unavailable baselines và sai khác reproduction; không thay thế âm thầm.

**Giới hạn nguồn context:** đã đọc file proposal gốc và toàn bộ nhận xét đính kèm. Link ChatGPT chia sẻ không trả nội dung qua công cụ web; thử Browser gặp lỗi môi trường. Vì vậy bản sửa chưa dựa trên các trao đổi chỉ xuất hiện trong link đó. Chưa chạy experiment hay xác nhận backend tương thích trong lần chỉnh proposal này.
