# Khảo sát: Quantization và Watermark/Fingerprint cho Diffusion Models
*Tổng hợp từ các paper NeurIPS / ICML / CVPR / ICCV / ECCV / NDSS / USENIX Security / IEEE TDSC / IEEE TIFS / WACV / ACSAC, tính đến 09/2026.*

---

## Tóm tắt điều hành

Khảo sát này trả lời hai câu hỏi nền tảng:

1. **Thế giới đang làm gì** ở giao điểm quantization ↔ watermark/fingerprint cho diffusion models?
2. Nếu watermark **sống sót tốt** dưới quantization thì cần đào sâu **hướng attack**; nếu **không sống sót** thì cần đào sâu **hướng defense**.

**Kết luận nhanh:** Trên gần như mọi paper watermark/fingerprint diffusion đã công bố số liệu robustness với "quantization" (Stable Signature, FingerInv, TrajPrint, Cert-LAS, AuthenLoRA, và cả các paper watermark DNN tổng quát như T2S, LineageMark), **watermark sống sót rất tốt** — nhưng **100% các thí nghiệm này đều dùng quantizer benign/chuẩn** (min-max rounding, TFLite/PyTorch PTQ mặc định, torchao preset, hoặc đơn giản là ép kiểu FP16/BF16). **Chưa có paper nào** kiểm tra một quantizer bị **điều khiển đối kháng** (adversarially steered) nhắm thẳng vào watermark của diffusion model, trong khi ở nhánh DNN classifier/LLM tổng quát, dòng nghiên cứu "malicious/quantization-conditioned backdoor" đã tồn tại từ 2021 và đang phát triển mạnh (đến tận NDSS 2026).

→ Theo đúng luật quyết định bạn đề ra: **watermark diffusion giữ tốt dưới quantization thường** ⇒ khảo sát này đào sâu **hướng ATTACK** (Phần V), vì đó là khoảng trống thật sự chưa được khai thác, không phải hướng defense (vốn chưa có gì để phòng thủ chống lại).

---

## Phần I — Quantization "thường" cho Diffusion Models (nền tảng kỹ thuật, không có security lens)

Đây là các paper thuần về hiệu năng/nén, dùng để hiểu **không gian tham số thật sự mà một PTQ pipeline hiện đại cho phép** (bit-width, calibration, timestep-awareness…) — tức là "Ψ_deploy" trông như thế nào trong thực tế.

| Paper | Venue | Đóng góp chính |
|---|---|---|
| **Q-Diffusion** (Li et al.) | ICCV 2023 | PTQ đầu tiên chuyên biệt cho diffusion; calibration theo timestep vì phân phối activation thay đổi mạnh qua các bước denoise. |
| **PTQD** (He et al.) | NeurIPS 2023 | Phân tích quantization noise trong diffusion models một cách hệ thống, tách sai số tương quan và không tương quan để bù trừ. |
| **MixDQ** (ECCV 2024) | ECCV 2024 | Metric-decoupled sensitivity analysis + mixed-precision search cho text-to-image diffusion. |
| **Q-DiT** (Chen et al.) | CVPR 2025 | PTQ cho Diffusion Transformers (DiT), khai thác biến thiên độ nhạy theo cả không gian lẫn thời gian (timestep). |
| **PQD** (Ye et al.) | WACV 2025 | Mở rộng PTQ diffusion sang độ phân giải cao và text-to-image, khắc phục hạn chế của các PTQ diffusion sớm hơn. |

**Nhận xét:** Toàn bộ dòng này coi quantizer là *benign, cố định, do người phòng vệ/nhà cung cấp mô hình chọn để tối ưu hiệu năng* — không ai coi quantizer là một đối tượng có thể bị kẻ tấn công điều khiển. Đây chính là giả định ngầm mà nhánh "malicious quantization" (Phần IV) thách thức.

---

## Phần II — Watermarking / Fingerprinting cho Diffusion Models: taxonomy

| Phương pháp | Venue | Cơ chế | Vị trí nhúng | Verification |
|---|---|---|---|---|
| WatermarkDM / *"A Recipe for Watermarking Diffusion Models"* (Zhao et al.) | arXiv 2023 (nhiều venue trích dẫn) | Backdoor-based: fine-tune trên dữ liệu đã watermark, trigger prompt/image | Toàn bộ model | Black-box (trigger response) |
| **Stable Signature** (Fernandez et al.) | ICCV 2023 | Fine-tune decoder LDM để nhúng chuỗi bit vào mọi ảnh sinh ra | Decoder (VAE) | Black-box (bit decoding) |
| **Tree-Ring Watermarks** (Wen et al.) | NeurIPS 2023 | Nhúng pattern vào Fourier space của initial noise; verify bằng DDIM inversion | Không sửa model — sửa initial latent | Cần truy cập inversion (semi white-box) |
| **AquaLoRA** (Feng et al.) | ICML 2024 | Watermark LoRA gắn vào UNet, 2-giai đoạn (latent watermark pretrain + prior-preserving fine-tune) | UNet (qua LoRA) | White-box |
| **Gaussian Shading** (Yang et al.) | CVPR 2024 | Watermark "provably lossless" bằng cách nhúng vào phân phối Gaussian của latent | Initial noise/latent | Semi white-box (inversion) |
| **ROBIN** (Huang et al.) | NeurIPS 2024 | Nhúng watermark mạnh ở trạng thái trung gian, dùng adversarial optimization để "giấu" watermark khỏi ảnh cuối | Sampling process | Inversion-based |
| **SleeperMark** (Wang et al.) | CVPR 2025 | Watermark bền vững qua fine-tuning (LoRA/DreamBooth/Custom Diffusion) cho T2I | UNet | Black-box (bit decoding) |
| **GaussMarker** (Li et al.) | ICML 2025 | Watermark "dual-domain" (kết hợp pixel + latent) | Latent + pixel | Semi white-box |
| **FingerInv** (Teng et al.) | CVPR 2025 | **Fingerprint nội tại (non-invasive)** — không sửa tham số; dùng "crossing route" qua performance border-zone để tạo latent code đặc trưng | Không sửa model | Black-box (QR-code scan) hoặc semi white-box (fingerprint-domain) |
| **DiffIP** (Li et al.) | ICCV 2025 | Representation fingerprint cho IP protection | Nội tại | — |
| **TrajPrint** / *"Lossless Copyright Protection via Intrinsic Model Fingerprinting"* (Chen et al.) | arXiv 2601.21252 | Fingerprint dựa trên quỹ đạo sampling, không sửa tham số | Nội tại | — |
| **Cert-LAS** (Qi, Li, Liang, Tu, Tao) | **ICML 2026** | **Certified MOV đầu tiên cho T2I diffusion** — layer-adaptive randomized smoothing + diffusion classifier làm watermark carrier, chứng minh certified radius qua paired t-test | UNet (trigger-free, qua diffusion classifier) | Black-box |
| **WAVES benchmark** (An et al.) | ICML 2024 | Benchmark chuẩn hoá robustness watermark ảnh — 26+ loại attack (nén, crop, regeneration, adversarial) | — | — |

**Điểm quan trọng về WAVES:** đây là benchmark uy tín nhất về robustness watermark ảnh, nhưng **toàn bộ 26+ attack đều ở mức image-space** (JPEG, resize, regeneration bằng VAE/diffusion, adversarial perturbation trên ảnh) — **không có attack nào ở mức model-weight quantization**. Đây là một khoảng trống có thể ghi nhận: literature benchmark hoá robustness ảnh rất kỹ, nhưng gần như bỏ qua trục "tấn công ở tầng deployment/model-compression".

---

## Phần III — Số liệu robustness dưới quantization đã công bố (tổng hợp)

Đây là phần cốt lõi để trả lời câu hỏi "giữ được hay không".

| Phương pháp | Loại quantizer test | Bit-width | Kết quả | Adversarial? |
|---|---|---|---|---|
| **Stable Signature** (Appendix B.3) | Naive min-max rounding (tự cài) | 8-bit, 4-bit | Bit accuracy **0.99** ở cả 8-bit và 4-bit | ❌ Benign |
| **FingerInv** (Table 2) | FP32→FP16 (SD, Pixart), FP32→BF16 (DeciDiffusion) | ~FP16/BF16 (gần lossless) | **100% success rate**, ngang bằng watermark xâm lấn | ❌ Benign, gần như không phải PTQ thật |
| **TrajPrint** | FP16, BFloat16 | FP16/BF16 | Bit-accuracy **>0.95** | ❌ Benign |
| **Cert-LAS** (Appendix E) | torchao (thư viện chuẩn) | W8A32, W8A8, W4A32 | VSR **=1.000** ở W8A32/W8A8; tụt ở W4A32 nhưng lúc đó DreamSim đã vượt ngưỡng dùng được | ❌ Benign |
| **AuthenLoRA** (LoRA watermark) | Model quantization (không nêu rõ chi tiết) | — | Robust, được báo cáo trong Fig. 6 của paper | ❌ Benign |
| **T2S** (watermark DNN classifier, không phải diffusion) | Standard weight quantization | 8-bit, 4-bit | WSR giữ **100%** kể cả ở 4-bit, dù accuracy giảm >5% | ❌ Benign |
| **LineageMark** (watermark DNN tổng quát) | Post-training quantization | 8-bit, 4-bit | 8-bit: **100%** trích xuất; 4-bit: vẫn **>90%** WER | ❌ Benign |

**Kết luận Phần III (rất nhất quán qua mọi domain — diffusion, classifier, LoRA):**
> **Watermark/fingerprint hiện tại sống sót cực tốt dưới quantization *benign*, kể cả xuống tới 4-bit** — nhưng **không một nghiên cứu nào** thử một quantizer được **tối ưu chủ động** để nhắm vào watermark trong khi vẫn giữ chất lượng sinh ảnh (điều mà PTQ nén ảnh thông thường không có động cơ để làm). Đây chính là gap "standard robustness ≠ worst-case robustness" mà chưa ai lấp đầy cho diffusion watermark.

---

## Phần IV — Malicious / Adversarial Quantization: dòng lịch sử (ngoài diffusion, vì trong diffusion gần như chưa có)

Vì malicious quantization **cho diffusion+watermark** gần như chưa tồn tại (đúng như bạn dự đoán — "hiếm"), phần này lấy nền từ dòng nghiên cứu malicious quantization tổng quát trên DNN/LLM, đã có ~5 năm lịch sử:

| Năm | Paper | Venue | Cơ chế |
|---|---|---|---|
| 2021 | **Hong et al., "Qu-anti-zation"** | NeurIPS 2021 | Khai thác quantization artifact để tạo outcome đối kháng — một trong những paper đặt nền móng đầu tiên. |
| 2021 | **Ma et al., "Quantization Backdoors to Deep Learning Commercial Frameworks"** | IEEE TDSC 2023 (arXiv 2021) | Model FP32 **hoàn toàn "sạch"** (4 backdoor-defense hàng đầu không phát hiện được), nhưng khi bị TFLite/PyTorch Mobile PTQ về INT8 theo pipeline PTQ **mặc định**, backdoor kích hoạt với ASR ~100%. |
| 2021 | **Pan et al., "Understanding the threats of trojaned quantized neural network in model supply chains"** | ACSAC 2021 | Phân tích threat model trojan quantized network trong chuỗi cung ứng model. |
| 2022 | **Tian et al., "Stealthy backdoors as compression artifacts"** | IEEE TIFS 2022 | Backdoor được thiết kế như "tác dụng phụ" của nén mô hình. |
| 2022 | **BppAttack** (Wang, Zhai, Ma) | CVPR 2022 | Dùng image quantization + contrastive learning để tạo trojan ẩn (khác lớp: quantize *ảnh* chứ không phải *trọng số*). |
| 2024 | **Li et al., "Nearest is not dearest"** | CVPR 2024 | **Defense** thực dụng đầu tiên chống Quantization-Conditioned Backdoors (QCB). |
| 2024 | **Egashira et al., "Exploiting LLM Quantization"** | NeurIPS 2024 | Model LLM benign ở full-precision, malicious sau quantization — đưa "malicious quantization" thành chủ đề an ninh chính thống cho LLM. |
| 2026 | **Chen et al., QuRA — "Rounding-Guided Backdoor Injection in Deep Learning Model Quantization"** | **NDSS 2026** | Chỉ tối ưu **hướng rounding** (không cần retrain) để kích hoạt/khuếch đại backdoor, giữ gần như nguyên clean accuracy — không cần huấn luyện lại, chỉ cần chọn rounding trong lúc quantize. |
| 2026 | **Yang, Tsai, Yu, QVec — "Quantization as a Malicious Task"** | arXiv 2606.20254 (06/2026) | **Defense**: coi hướng dịch chuyển tham số do quantization gây ra như một "task vector" và triệt tiêu nó bằng task arithmetic, không cần retrain, không cần trigger sample. |

**Cấu trúc chung của tấn công (QCB — Quantization-Conditioned Backdoor):** huấn luyện/điều chỉnh model sao cho hành vi độc hại **ngủ đông** ở full-precision (qua được mọi kiểm tra an ninh ở FP32/FP16) và chỉ **"thức dậy"** sau khi bị đưa qua một PTQ pipeline cụ thể — vì bản thân phép làm tròn (rounding) tới lưới quantization tạo ra đúng vector dịch chuyển cần thiết để kích hoạt hành vi ẩn.

**Khoảng trống rõ ràng nhất:** Toàn bộ dòng QCB/QuRA/Egashira nhắm vào *tạo ra hành vi độc hại mới* (backdoor, jailbreak) khi quantize. Chưa có paper nào lật ngược logic đó để **phá hủy một tín hiệu đã tồn tại (watermark/fingerprint sở hữu)** bằng cách chọn quantizer một cách có chủ đích, *đặc biệt* trong bối cảnh diffusion model — nơi không gian tham số PTQ phong phú hơn nhiều (clipping, scale/zero-point, calibration set, và đặc biệt là **calibration theo timestep**, thứ hoàn toàn không tồn tại trong LLM/classifier).

---

## Phần V — Đào sâu: hướng ATTACK (vì Phần III cho thấy watermark giữ tốt dưới quantization thường)

Theo đúng logic bạn đặt ra, vì Phần III cho thấy **watermark/fingerprint hiện tại sống sót rất tốt** dưới mọi quantizer benign đã test, hướng cần đào sâu là **attack**, không phải defense (chưa có threat thật để phòng thủ chống lại).

### V.1 — Vì sao quantization "thường" không phá được watermark

Nhìn qua dữ liệu Phần III, có 3 lý do lặp lại:
1. **Sai số quantization benign là "vô hướng"** (undirected) — GPTQ/round-to-nearest tối thiểu hoá sai số tái tạo *trung bình*, không có động cơ nào để nó trùng với hướng làm hỏng watermark cụ thể.
2. **Watermark hiện đại được train để chịu nhiễu ngẫu nhiên** (augmentation trong lúc train, hoặc — với Cert-LAS — smoothing chủ động), nên một sai số không định hướng dễ bị "trung hoà".
3. **Các bit-width benign phổ biến (W8, thậm chí W4A32)** vẫn còn margin khá lớn trước khi ảnh hưởng tới chất lượng sinh ảnh — nghĩa là còn "chỗ trống" trong ngân sách chất lượng mà một attacker có chủ đích có thể khai thác mà chưa ai khai thác.

### V.2 — Không gian tấn công còn bỏ ngỏ cho diffusion cụ thể

So với LLM/classifier (nơi QCB/QuRA hoạt động), diffusion model PTQ có **nhiều bậc tự do hơn** mà một attacker có thể điều khiển — đây chính là "bề mặt tấn công" chưa ai chạm tới:

| Bậc tự do trong PTQ diffusion | Có trong LLM/classifier PTQ không? | Có bị khai thác bởi QCB/QuRA/Egashira chưa? |
|---|---|---|
| Bit-width, group size, scale/zero-point | Có | Một phần (QuRA chỉ khai thác rounding) |
| Clipping threshold | Có | Chưa |
| Calibration-set reweighting | Có (yếu hơn) | Chưa |
| **Calibration theo timestep** (timestep-weighted calibration) | **Không tồn tại** | Chưa (không thể, vì không có timestep) |
| Layer × timestep sensitivity (Q-Diffusion, MixDQ, Q-DiT đã đo, nhưng chỉ để *tối ưu chất lượng*) | Không tồn tại | Chưa ai *steer* nó để phá watermark |

### V.3 — Các câu hỏi nghiên cứu cụ thể còn mở (để đào sâu tiếp)

1. **Liệu một quantizer được tối ưu có chủ đích** (chọn calibration set + clipping + timestep-weighting để cực đại hoá thiệt hại lên watermark, trong khi ràng buộc suy giảm chất lượng sinh ảnh trong một ngân sách ε) **có phá được** Stable Signature / AquaLoRA / FingerInv / Cert-LAS **ở cùng bit-width** mà benign PTQ đã chứng minh là an toàn (W8, thậm chí W4)?
2. **Cert-LAS cụ thể**: certified radius của họ được chứng minh cho nhiễu Gaussian/Mahalanobis-ball (Theorem 4.9), không phải cho tập sai số quantization-realizable. Liệu norm Mahalanobis của một sai số quantization bị steer có chủ đích có vượt certified radius đã công bố hay không — đây là câu hỏi thực nghiệm cụ thể, có thể trả lời trực tiếp bằng cách đo ||e_ψ*||_σk và so với R* họ công bố ở Table 2.
3. **Tính chuyển giao (transferability)**: các kỹ thuật QCB/QuRA hiện tại đều giả định attacker biết chính xác backdoor/watermark cụ thể đang nhắm tới. Với watermark diffusion (mỗi bản deploy có secret/key riêng — AquaLoRA key, hay fingerprint riêng như FingerInv), câu hỏi mở là: một quantizer được tối ưu trên các **surrogate watermark instance** có tổng quát hoá được sang một **victim instance với secret chưa từng thấy** hay không? Đây là câu hỏi chưa ai trả lời cả ở nhánh QCB (thường single-target) lẫn nhánh watermark-robustness (thường chỉ test benign).
4. **Deployment validity**: mọi kỹ thuật QCB hiện có (Ma et al., QuRA) đều xuất ra checkpoint **thực sự load được** trên backend PTQ chuẩn (TFLite, PyTorch Mobile). Bất kỳ hướng tấn công watermark-quantization nào cũng cần đạt cùng tiêu chuẩn đó để có ý nghĩa an ninh thực tế, chứ không chỉ là một continuous relaxation trên giấy.

### V.4 — Vì sao đây là attack đáng làm (không phải chỉ để "thắng" mà còn để bảo vệ)

Theo đúng chuẩn mực đã thấy ở toàn bộ dòng QCB/Egashira/QuRA/Cert-LAS: mỗi paper attack đều đi kèm defense hoặc lời kêu gọi defense (Egashira → chưa có defense hoàn chỉnh; QuRA → không đề xuất defense riêng nhưng mở hướng cho QVec-style; Cert-LAS → certified). Một nghiên cứu tấn công watermark-quantization cho diffusion, nếu làm, nên đi kèm:
- Responsible disclosure cho các thư viện watermark mã nguồn mở bị ảnh hưởng (AquaLoRA, Stable-Signature).
- Một defense đi kèm (ví dụ: mở rộng QVec-style task-arithmetic correction, hoặc mở rộng certified radius của Cert-LAS sang tập quantization-realizable thay vì chỉ Gaussian ball).

---

## Phần VI — Danh mục tài liệu tham khảo

**Quantization cho diffusion (benign/hiệu năng):**
- Li et al., *Q-Diffusion: Quantizing Diffusion Models*, ICCV 2023
- He et al., *PTQD: Accurate Post-Training Quantization for Diffusion Models*, NeurIPS 2023
- MixDQ, ECCV 2024
- Chen et al., *Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers*, CVPR 2025
- Ye et al., *PQD: Post-training Quantization for Efficient Diffusion Models*, WACV 2025

**Watermarking/Fingerprinting cho diffusion:**
- Zhao et al., *A Recipe for Watermarking Diffusion Models*, arXiv 2023
- Fernandez et al., *The Stable Signature: Rooting Watermarks in Latent Diffusion Models*, ICCV 2023
- Wen et al., *Tree-Rings Watermarks*, NeurIPS 2023
- Feng et al., *AquaLoRA: Toward White-box Protection for Customized Stable Diffusion Models via Watermark LoRA*, ICML 2024
- Yang et al., *Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models*, CVPR 2024
- Huang, Wu, Wang, *ROBIN: Robust and Invisible Watermarks for Diffusion Models with Adversarial Optimization*, NeurIPS 2024
- Wang et al., *SleeperMark: Towards Robust Watermark against Fine-Tuning Text-to-image Diffusion Models*, CVPR 2025
- Li et al., *GaussMarker: Robust Dual-Domain Watermark for Diffusion Models*, ICML 2025
- Teng, Quan, Wang, Huang, Ji, *Fingerprinting Denoising Diffusion Probabilistic Models* (FingerInv), CVPR 2025
- Li et al., *DiffIP: Representation Fingerprints for Robust IP Protection of Diffusion Models*, ICCV 2025
- Chen et al., *Lossless Copyright Protection via Intrinsic Model Fingerprinting* (TrajPrint), arXiv 2601.21252
- Qi, Li, Liang, Tu, Tao, *Cert-LAS: Toward Certified Model Ownership Verification for Text-to-Image Diffusion Models via Layer-Adaptive Smoothing*, ICML 2026
- An et al., *WAVES: Benchmarking the Robustness of Image Watermarks*, ICML 2024

**Watermark DNN tổng quát (không phải diffusion, nhưng có số liệu quantization hữu ích):**
- Uchida et al., *Embedding Watermarks into Deep Neural Networks*, 2017
- Nagai et al., *Digital Watermarking for Deep Neural Networks*, 2018
- T2S (rehearsal-based extraction-resistant watermarking), arXiv 2606.11698
- LineageMark, arXiv 2606.17123
- AuthenLoRA, arXiv 2511.21216

**Malicious/Adversarial Quantization:**
- Hong, Panaitescu-Liess, Kaya, Dumitras, *Qu-anti-zation: Exploiting Quantization Artifacts for Achieving Adversarial Outcomes*, NeurIPS 2021
- Ma et al., *Quantization Backdoors to Deep Learning Commercial Frameworks*, IEEE TDSC 2023 (arXiv 2108.09187)
- Pan et al., *Understanding the Threats of Trojaned Quantized Neural Network in Model Supply Chains*, ACSAC 2021
- Tian, Suya, Xu, Evans, *Stealthy Backdoors as Compression Artifacts*, IEEE TIFS 2022
- Li et al., *Nearest is not Dearest: Towards Practical Defense against Quantization-Conditioned Backdoor Attacks*, CVPR 2024
- Egashira et al., *Exploiting LLM Quantization*, NeurIPS 2024
- Chen et al., *QuRA: Rounding-Guided Backdoor Injection in Deep Learning Model Quantization*, NDSS 2026
- Yang, Tsai, Yu, *Quantization as a Malicious Task: Removing Quantization-Conditioned Backdoors via Task Arithmetic* (QVec), arXiv 2606.20254

---

*Ghi chú phương pháp luận: khảo sát này được tổng hợp qua tra cứu web tính đến 09/2026, ưu tiên paper có thể xác minh (venue, DOI/arXiv ID cụ thể). Đây không phải một systematic review đầy đủ — một số nhánh nhỏ hơn (ví dụ watermark cho diffusion audio/video, hoặc quantization cho non-U-Net architecture) chưa được bao phủ.*
