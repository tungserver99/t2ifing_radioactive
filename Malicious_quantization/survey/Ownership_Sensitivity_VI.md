# Ownership Sensitivity trong Diffusion Model Quantization

## 1. Định nghĩa

Độ nhạy quyền sở hữu (Ownership Sensitivity) đo mức độ ảnh hưởng của quá trình lượng tử hóa (Post-Training Quantization - PTQ) đến khả năng xác minh quyền sở hữu của mô hình diffusion.

Với layer \(l\) và cấu hình quantization \(q\):

\[
S^{own}_{l,q}
=
\Delta V(Q_{l,q}(\theta))
\]

Trong đó:

- \(Q_{l,q}(\theta)\): mô hình sau khi lượng tử hóa layer \(l\) với cấu hình \(q\).
- \(V(\cdot)\): hàm xác minh quyền sở hữu.
- \(\Delta V\): mức suy giảm của tín hiệu xác minh.

---

## 2. Ý nghĩa

Ownership Sensitivity đo ảnh hưởng của quantization đến:

- độ chính xác watermark:

\[
\mathrm{Watermark\ Accuracy}
\]

- độ tin cậy fingerprint:

\[
\mathrm{Fingerprint\ Confidence}
\]

Layer có \(S^{own}_{l,q}\) lớn nghĩa là layer đó rất nhạy với quantization về mặt ownership.

---

# 3. Generation Sensitivity

Tương tự, định nghĩa độ nhạy sinh ảnh:

\[
S^{gen}_{l,q}
=
\Delta U(Q_{l,q}(\theta))
\]

Trong đó \(U(\cdot)\) biểu diễn utility của diffusion model:

- FID;
- CLIP score;
- LPIPS;
- reconstruction error.

---

# 4. Ownership-Generation Sensitivity Mismatch

Giả thuyết trung tâm:

\[
\mathrm{Rank}(S^{gen})
\neq
\mathrm{Rank}(S^{own})
\]

Điều này có nghĩa:

> Những layer quan trọng để giữ chất lượng sinh ảnh không nhất thiết là những layer quan trọng để giữ bằng chứng quyền sở hữu.

Ví dụ:

| Layer | Generation Sensitivity | Ownership Sensitivity |
|---|---|---|
| Attention block A | Low | High |
| Convolution block B | High | Low |

Attention block A có thể được lượng tử hóa mà ảnh vẫn giữ chất lượng, nhưng watermark/fingerprint có thể suy giảm mạnh.

---

# 5. Ownership-Utility Mismatch Score

Định nghĩa:

\[
M_{l,q}
=
\frac{
S^{own}_{l,q}
}
{
S^{gen}_{l,q}+\epsilon
}
\]

Trong đó:

- \(M_{l,q}\) lớn:
  - ownership rất nhạy;
  - generation ít bị ảnh hưởng.

Các vùng này là mục tiêu quan trọng của Ownership-Aware Quantization.

---

# 6. Ownership Subspace

Giả sử tham số diffusion model:

\[
\theta
\]

gồm hai thành phần:

\[
\theta
=
\theta_{gen}
+
\theta_{own}
\]

Trong đó:

- \(\theta_{gen}\): thông tin phục vụ sinh ảnh.
- \(\theta_{own}\): thông tin phục vụ xác minh ownership.

Quantization tạo ra sai số:

\[
e_q
=
Q(\theta)-\theta
\]

Không chỉ độ lớn sai số quan trọng, mà hướng của sai số so với ownership information cũng quan trọng.

Xác định ownership direction:

\[
v_{own}
=
\nabla_{\theta}V(\theta)
\]

---

# 7. Quantization Ownership Alignment

Định nghĩa:

\[
A_q
=
\frac{
\langle e_q,v_{own}\rangle
}
{
||e_q||\,||v_{own}||
}
\]

Ý nghĩa:

Nếu:

\[
A_q \rightarrow 1
\]

quantization noise cùng hướng với ownership signal và có khả năng làm suy giảm verification.

Nếu:

\[
A_q \rightarrow 0
\]

quantization chủ yếu ảnh hưởng vùng không liên quan đến ownership.

---

# 8. Research Insight

Thay vì hỏi:

> INT4 có làm watermark giảm không?

Câu hỏi sâu hơn là:

> Quantization error tác động vào vùng thông tin nào của diffusion model?

Mục tiêu là tìm ra:

\[
\boxed{
\text{Generation Information}
\neq
\text{Ownership Information}
}
\]

và sử dụng hiểu biết này để thiết kế quantization bảo toàn quyền sở hữu.
