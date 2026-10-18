# Hyper-heuristic cho tìm kiếm W4 trên SleeperMark: khảo sát và thiết kế kiểm chứng

Ngày cập nhật: 30/09/2026. Thiết kế đã được triển khai dưới các tên `behavior_archive_genetic_w4`, `bandit_hyperheuristic_w4`, `llm_hyperheuristic_w4` và đối chứng `public_trigger_bandit_w4`; **chưa có kết quả attack** cho các nhánh mới. Phạm vi chỉ có SleeperMark. Giữ hai chế độ tách biệt: blind model-only và đối chứng dùng trigger công khai; không đưa key, extractor hay TEST vào vòng tìm kiếm.

## 1. Khoảng trống trong code hiện tại

`wmq_evolution.py` đã có gene scale/rounding theo nhóm UNet, mutation/crossover theo cặp, bốn operator với EMA credit, và descriptor hành vi từ noise-prediction. `wmq_llm_proposals.py` cho Qwen2.5-3B đề xuất trực tiếp ba chromosome mỗi thế hệ. Đây là **LLM làm candidate generator**; nó chưa thiết kế/chọn chính sách tìm kiếm. Trong `search.population_order`, khi dùng `quality_ratio`, `quality_order` chỉ nhìn `(ordinary_loss, proxy_loss)` và không dùng descriptor. Do đó cơ chế diversity hiện tại có thể bị vô hiệu chính ở cấu hình quality-corridor đang chạy.

Kết quả local của suite 27/09 cho thấy `adaptive_genetic_w4` BA 99,08%, `llm_genetic_w4` BA 98,94%, `local_proposal_genetic_w4` BA 99,00%; cả ba TPR 100/100 trên TEST. Sai khác nhỏ về BA chưa chứng minh LLM hữu ích; không có evidence proxy hiện tại liên hệ đủ mạnh với detector để đạt BA 0,5–0,6. Cần đánh giá suite mới đã chạy xong trước khi chốt ưu tiên.

## 2. Học được gì từ các công trình của nhóm Huỳnh Thị Thanh Bình

| Công trình | Cơ chế trong paper | Chuyển sang bài toán này |
| --- | --- | --- |
| [HSEvo, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34898) | Flash reflection tổng hợp cặp tốt/xấu theo thế hệ; Harmony Search tinh chỉnh elite; theo dõi diversity. | Một lần phản tỉnh trên **thống kê FIT của quần thể** mỗi vài thế hệ, rồi phân bổ ngân sách cho local refinement hoặc exploration. Mọi offspring vẫn phải qua hard-W4 evaluation. |
| [MPaGE, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/download/41024/44985) | Archive/grid trong objective space, chọn đại diện và lai giữa các nhóm khác biệt. | Giữ nhiều niche trong hành lang quality, tránh quần thể sụp về các genome có cùng hành vi. Không dùng code embedding: cá thể ở đây là quantizer, nên dùng descriptor phản ứng UNet. |
| [ReVEL, arXiv v3, 2026](https://arxiv.org/html/2604.04940v3) | Nhóm heuristic theo hành vi; multi-turn reflection và search memory qua thế hệ. | Ghi lại FIT failures/successes theo nhóm prompt và timestep. LLM chọn **policy có giới hạn**, không suy luận ownership từ proxy. Paper thử heuristic programs trên COP, không thử watermark hay quantization. |
| [MF-LTGA, Information Sciences 2020](https://arxiv.org/abs/2005.03090) | Học linkage giữa biến, dùng linkage tree cho crossover. | Thử block linkage của cặp scale–rounding. Với 12 cá thể/thế hệ, ước lượng mutual information 7 giá trị/gene rất nhiễu; cần gom archive nhiều thế hệ và đối chứng nhóm ngẫu nhiên. |

Không nên chép nguyên LLM sinh code và chạy code như HSEvo/ReVEL: tăng bề mặt lỗi, khó tái lập, và LLM 3B không có bằng chứng suy ra cấu trúc owner từ các loss hiện có. Adaptation khả thi hơn là LLM **chọn heuristic tìm kiếm** trong DSL hữu hạn.

## 3. Phương pháp đề xuất: behavior-aware hyper-heuristic W4

### Tầng dưới: archive và offspring

Mỗi cá thể `g` lưu 18 gene, ordinary noise loss trên FIT, proxy loss trên FIT, descriptor `b(g)` theo prompt/timestep và số UNet forwards. Các phép biến đổi hợp lệ:

1. Mutation một tọa độ; mutation nguyên cặp scale–rounding.
2. Crossover theo attention group cố định; đối chứng crossover theo group ngẫu nhiên cùng kích thước.
3. Harmony-style local move quanh một elite **chỉ tính là offspring**, không có bước thử miễn phí.
4. Immigrant thưa gần RTN; giữ xác suất khám phá tối thiểu.

Archive có anchor RTN và các ô đã cố định trước theo `ordinary_loss / ordinary_RTN` và proxy gain. Trong một ô, giữ ứng viên có FIT score tốt và descriptor khác nhau; fitness vẫn quality-first. Chọn cha mẹ xen kẽ cùng niche và khác niche. Cần làm ablation archive theo descriptor so với chỉ sắp xếp proxy; nếu diversity không cải thiện SELECT, bỏ cơ chế này.

### Tầng trên: chính sách chọn operator

Thay LLM sinh gene bằng một chính sách JSON bị giới hạn: tỷ lệ ngân sách dành cho bốn operator, độ lớn bước mutation thuộc `{1,2}`, tỷ lệ mating cùng/khác niche, và tỷ lệ local refine. Đầu vào chỉ là báo cáo FIT đã rút gọn: số ứng viên quality-feasible, FIT proxy gain, khoảng cách hành vi giữa các niche, tỷ lệ duplicate, reward của operator trên *offspring mới*. Không cung cấp owner report, TEST, SELECT, prompt string hay tensor model. Xác thực schema, chuẩn hóa xác suất, và fallback sang chính sách xác định khi JSON lỗi. Không thực thi code LLM sinh ra.

Gọi LLM một lần mỗi 3 thế hệ để chính sách được dùng cho nhiều offspring. Lưu cả input, output, model revision, tokens, thời gian, policy hiệu lực và lý do fallback. `reward` cho operator phải xét **cải thiện best feasible proxy trên FIT**, trừ duplicate và có exploration floor; không chỉ so offspring với một parent yếu. Có thể dùng contextual bandit/UCB xác định làm đối chứng mạnh cho LLM với cùng thông tin và cùng 156 hard-W4 evaluations. LLM chỉ có giá trị nếu vượt bandit/GA ở SELECT và owner evaluation sau freeze, không chỉ ở FIT.

### Hai chế độ thông tin

- **Blind:** tất cả feedback trên prompt thường, không có trigger, key hoặc extractor. Descriptor chỉ mô tả phản ứng noise-prediction, không gọi nó là ownership signal.
- **Public-trigger control:** thêm loss và descriptor trên trigger `*[Z]& ` đã công bố, vẫn không dùng key/extractor. Báo cáo riêng; không trộn kết quả hay claim model-only. Giữ ordinary-prompt quality constraint và đo cả chất lượng ảnh triggered.

## 4. Thí nghiệm quyết định

Chạy cùng checkpoint, prompt split, seed, số lượng hard-W4 evaluations, SELECT budget và owner TEST: (A) GA hiện tại; (B) GA + quality-diversity archive; (C) B + deterministic contextual bandit; (D) B + bounded LLM policy; (E) LLM trực tiếp sinh genome hiện tại. Thêm random linkage, no-memory và no-reflection ablation. Đo FIT, SELECT, số genome feasible, coverage/diversity, duplicate rate, wall time/UNet forwards, rồi mới đo BA/TPR, FID và triggered/ordinary SSIM trên TEST sau khi freeze.

Quy tắc quyết định: nếu (D) chỉ thắng FIT hoặc chậm hơn rõ mà không thắng (C) trên SELECT, bỏ LLM policy. Nếu mọi phương án SELECT tốt hơn nhưng BA/TPR vẫn không đổi, nút thắt là **proxy không chuyển sang watermark**, không phải optimizer. Khi đó cần nghiên cứu tín hiệu quan sát được mạnh hơn hoặc giới hạn kết luận về attack blind; tăng generations/LLM reasoning không tự sửa được sai lệch mục tiêu.

Các kết quả TEST cũ đã được xem nhiều lần để định hướng phương pháp, nên là development evidence. Claim cuối cần prompt/seed/checkpoint holdout mới, chốt policy và hyperparameters trước khi mở owner evaluation. Nếu có nhiều key hoặc checkpoint, phân tích độ ổn định qua chúng; một key một lần chạy không đủ chứng minh attack tổng quát.

## 5. Thứ tự thực hiện

1. Sửa quality-corridor survival để descriptor thực sự tham gia; so với sắp xếp hiện tại và kiểm tra quality không tụt.
2. Thêm deterministic archive + bandit, tách thành nhánh, giữ budget bằng nhau.
3. Chỉ thêm bounded LLM policy khi bandit cho thấy search policy có headroom. Không đưa nhánh mới vào mặc định trước khi có SELECT và owner evidence.

Đây là một **phương pháp tối ưu tìm kiếm**, chưa phải phương pháp đo watermark. Novelty tiềm năng chỉ mạnh nếu cho thấy behavior-aware search cải thiện attack ở cùng ràng buộc và ngân sách, trên dữ liệu đánh giá mới. Việc kết hợp HSEvo/MPaGE/ReVEL theo tên không tự tạo novelty.
