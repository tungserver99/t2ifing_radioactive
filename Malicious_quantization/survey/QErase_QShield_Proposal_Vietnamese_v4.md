# QErase: Ownership-Selective Malicious Quantization for Diffusion Models

## QShield: Adversarially Quantization-Robust Ownership Embedding

## 0. Tổng quan ý tưởng

### Câu hỏi nghiên cứu

Liệu một diffusion model có watermark hoặc fingerprint bền vững với các
phương pháp Post-Training Quantization (PTQ) thông thường có còn an toàn
trước một quantizer được lựa chọn có chủ đích?

Giả thuyết trung tâm:

    Standard quantization robustness
    does not imply
    worst-case malicious quantization robustness

QErase xem quantizer như một đối tượng tấn công.

Attacker không sửa trực tiếp trọng số mô hình. Thay vào đó, attacker lựa
chọn một cấu hình quantization hợp lệ trong phạm vi deployment thực tế
để tạo ra sai số quantization có chủ đích.

Mục tiêu:

-   làm suy giảm ownership verification;
-   duy trì chất lượng sinh ảnh;
-   tạo checkpoint INT4/INT8 có thể triển khai.

------------------------------------------------------------------------

# 1. Novelty

## Không claim

QErase không claim:

-   phương pháp malicious quantization đầu tiên;
-   phương pháp loại bỏ watermark diffusion đầu tiên.

Các hướng này đã tồn tại.

## Claim chính

Đóng góp của QErase là:

> Lần đầu xem tập sai số quantization có thể sinh ra từ một pipeline PTQ
> thực tế như không gian hành động của attacker trong bài toán bảo vệ
> quyền sở hữu diffusion model.

Khác biệt:

  Phương pháp     Không gian tấn công
  --------------- ------------------------------------------------
  Weight attack   sửa weight tùy ý
  Pixel attack    sửa ảnh hoặc latent
  Standard PTQ    quantizer cố định
  QErase          lựa chọn quantizer hợp lệ để điều khiển sai số

------------------------------------------------------------------------

# 2. Threat Model

## 2.1 Quantizer adversarial

Một quantizer được biểu diễn:

    Q_psi

với:

    psi belongs to Psi_deploy

Trong đó Psi_deploy gồm các lựa chọn hợp lệ:

-   weight bit-width;
-   activation bit-width;
-   group size;
-   scale;
-   zero-point;
-   clipping threshold;
-   calibration distribution;
-   timestep-aware calibration;
-   rounding strategy.

------------------------------------------------------------------------

## 2.2 Giới hạn attacker

Được phép:

-   lựa chọn cấu hình PTQ;
-   tối ưu tham số quantization;
-   sử dụng surrogate ownership models.

Không được phép:

-   thay đổi weight ngoài quá trình quantization;
-   fine-tune victim;
-   biết secret/key của victim;
-   truy cập verifier của victim.

------------------------------------------------------------------------

# 3. Quantization-Realizable Error Set (QRES)

Sai số do quantization tạo ra:

    E_Q(theta) = { Q_psi(theta) - theta | psi in Psi_deploy }

QRES biểu diễn toàn bộ perturbation mà một deployment quantizer hợp lệ
có thể tạo ra.

Khác với adversarial weight perturbation:

    Delta theta is arbitrary

QErase chỉ cho phép:

    e belongs to E_Q(theta)

------------------------------------------------------------------------

# 4. Vulnerability Formulation

Một model vulnerable nếu tồn tại:

    e in E_Q(theta)

thỏa hai điều kiện:

Generation không bị ảnh hưởng:

    ||J_gen * e|| <= epsilon

Ownership bị phá:

    g_own^T * e << 0

Nghĩa là:

-   ảnh vẫn sinh bình thường;
-   fingerprint/watermark giảm mạnh.

------------------------------------------------------------------------

# 5. QErase Attack

## Ownership-Selective Malicious Quantizer (OS-MQ)

OS-MQ gồm 4 module.

------------------------------------------------------------------------

## Module 1: Trajectory Ownership Profiler

Tính hai hướng nhạy cảm:

Generation direction:

    J_gen(l,t)

Ownership direction:

    g_own(l,t,m)

Mục tiêu tìm vùng:

    ownership sensitive
    +
    generation insensitive

theo layer và timestep.

------------------------------------------------------------------------

## Module 2: Quantization Error Steering

Tối ưu quantizer:

    psi_star =
    argmax(
        ownership damage
        -
        lambda * generation degradation
    )

Cụ thể:

    maximize:

    -g_own^T e_psi - lambda ||J_gen e_psi||^2

    where:

    e_psi = Q_psi(theta) - theta

Điểm mới:

Không chỉ đo sensitivity.

QErase chủ động điều khiển hướng của quantization error.

------------------------------------------------------------------------

## Module 3: Deployment Projection

Kết quả tối ưu liên tục được chuyển thành checkpoint thật:

-   INT4;
-   INT8;
-   backend compatible.

Projection giữ hai mục tiêu:

    standard PTQ similarity
    +
    malicious ownership steering

Có trade-off:

-   attack strength;
-   stealth.

------------------------------------------------------------------------

## Module 4: Secret/Fingerprint Agnostic Transfer

Mục tiêu:

kiểm tra attack có cần biết victim hay không.

Quy trình:

1.  Tạo K surrogate ownership models.
2.  Học một quantizer chung.
3.  Freeze quantizer.
4.  Áp dụng lên victim chưa từng thấy.
5.  Đánh giá ownership verification.

Hai nhóm tách riêng:

### Embedded watermark

Ví dụ:

-   AquaLoRA;
-   Stable Signature.

Đánh giá:

unseen key transfer.

### Intrinsic fingerprint

Ví dụ:

-   FingerInv;
-   TrajPrint.

Đánh giá:

unseen fingerprint model transfer.

------------------------------------------------------------------------

# 6. Metric

## Attack Budget Curve

Thay vì chỉ dùng một chỉ số:

    A_Q(epsilon)

được định nghĩa:

    maximize:

    ownership damage

    subject to:

    generation degradation <= epsilon

Ý nghĩa:

Trong một mức suy giảm chất lượng cho phép, malicious quantizer có thể
phá ownership đến đâu.

------------------------------------------------------------------------

# 7. QShield Defense

QShield huấn luyện watermark chịu được quantizer xấu nhất.

Mục tiêu:

    maximize watermark robustness

    against

    worst deployment-realizable quantizer

Không tối ưu cho một cấu hình INT8 hoặc INT4 cố định.

Certification chỉ được claim nếu có chứng minh toán học đầy đủ.

Nếu không:

QShield được báo cáo là empirical robust defense.

------------------------------------------------------------------------

# 8. Experimental Plan

## Model

-   Stable Diffusion 1.5;
-   SDXL;
-   Diffusion Transformer nếu đủ tài nguyên.

## Baseline

So sánh:

1.  Fixed PTQ.
2.  Random adaptive PTQ.
3.  Sensitivity-search PTQ.
4.  QuRA-style rounding-only attack.
5.  OS-MQ không dùng timestep weighting.
6.  Full QErase.

------------------------------------------------------------------------

# 9. Ba thí nghiệm bắt buộc

## Experiment 1

QErase phải vượt:

-   sensitivity search;
-   QuRA-style attack.

Cùng:

-   bit-width;
-   utility budget;
-   search budget.

------------------------------------------------------------------------

## Experiment 2

Transfer attack.

Quantizer học từ surrogate phải phá được:

-   unseen key;
-   unseen fingerprint model.

------------------------------------------------------------------------

## Experiment 3

Utility preservation.

Đánh giá:

-   FID;
-   KID;
-   CLIP score;
-   DreamSim.

------------------------------------------------------------------------

# 10. Contributions

## C1 - Threat model và formulation

Đề xuất:

Deployment-realizable malicious quantization cho diffusion ownership
verification.

Bao gồm:

-   QRES;
-   generation/ownership directions.

------------------------------------------------------------------------

## C2 - QErase / OS-MQ

Framework điều khiển quantization error:

-   ownership-selective steering;
-   diffusion trajectory awareness;
-   deployable quantization projection.

------------------------------------------------------------------------

## C3 - Transferability finding

Chứng minh malicious quantizer có thể chuyển từ surrogate sang victim mà
không cần victim secret.

------------------------------------------------------------------------

## C4 - QShield

Defense framework giúp watermark/fingerprint bền vững trước malicious
quantization.

------------------------------------------------------------------------

# 11. Khả năng CVPR

Ý tưởng có tiềm năng CVPR/ICCV/NeurIPS nếu đạt:

1.  QErase vượt QuRA-style baseline.
2.  Transfer sang unseen victim thành công.
3.  Ownership giảm mạnh nhưng generation quality giữ nguyên.

Contribution cốt lõi:

    Malicious but deployment-valid quantization
    can selectively erase ownership signals
    while preserving diffusion generation quality.
