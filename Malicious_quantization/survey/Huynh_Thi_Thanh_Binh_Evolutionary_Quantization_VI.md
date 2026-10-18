# Survey chọn lọc các bài của Huỳnh Thị Thanh Bình và khả năng áp dụng vào quantization

**Cập nhật 30/09/2026:** Các đề xuất paired crossover, adaptive operators và descriptor hành vi bên dưới đã có trong code SleeperMark sau ngày khảo sát gốc. Bản thiết kế hyper-heuristic tiếp theo và các đối chứng cần chạy nằm ở [SleeperMark_Hyperheuristic_LLM_Evolution_VI.md](SleeperMark_Hyperheuristic_LLM_Evolution_VI.md). Trạng thái “chưa triển khai” trong nội dung 27/09 chỉ phản ánh thời điểm viết bản gốc.

Ngày khảo sát: 27/09/2026. Phạm vi: các công trình tiến hóa, đa nhiệm và đa mục
tiêu có cơ chế phù hợp với repository hiện tại; không phải danh mục toàn bộ
công bố. Danh tính/đơn vị được đối chiếu từ [trang SoICT HUST](https://soict.hust.edu.vn/en/prof-huynh-thi-thanh-binh.html).
Các phần “áp dụng” dưới đây là đề xuất của survey, chưa được các paper chứng
minh trên Stable Signature/SleeperMark, và chưa được triển khai bởi lần survey này.

## 1. Các bài phù hợp nhất

### A. A Multifactorial Optimization Paradigm for Linkage Tree Genetic Algorithm

Huynh Thi Thanh Binh và cộng sự; Information Sciences 540 (2020), 325–344.
[Paper và metadata](https://arxiv.org/abs/2005.03090),
[full text](https://arxiv.org/html/2005.03090),
[DOI](https://doi.org/10.1016/j.ins.2020.05.132).

**Trong paper:** MF-LTGA học quan hệ giữa biến, tạo linkage tree riêng cho task,
và dùng nhóm biến làm mặt nạ crossover. Thử nghiệm gồm clustered shortest-path
tree và deceptive trap. Đây không phải paper về quantization.

**Đề xuất áp dụng — ưu tiên 1:** hiện `wmq_evolution.py` lai ghép độc lập từng
gene. Trong `wmq_sleeper_evolution.py`, một attention module có cặp gene scale
và rounding bias. Tách hai gene khi crossover có thể phá một phối hợp tốt.
Trước hết thử crossover theo cặp, rồi theo attention block; sau đó mới học
nhóm liên block từ dữ liệu search. Không gọi crossover theo cấu trúc có sẵn
là learned linkage tree.

Population 12 hiện tại quá ít để tin một ma trận mutual information phức tạp
giữa các biến có 7 giá trị. Nếu học linkage, cần archive đủ lớn, regularization
và kiểm tra độ ổn định; không suy quan hệ nhân quả từ tương quan trong elite.

**Ablation:** uniform crossover / cặp scale–bias / nhóm ngẫu nhiên cùng kích
thước / nhóm học được. Giữ nguyên objective, ngân sách UNet và rule chọn.

### B. Ensemble Multifactorial Evolution with Biased Skill-Factor Inheritance for Many-task Optimization

Huynh Thi Thanh Binh, Le Van Cuong, Ta Bao Thang, Nguyen Hoang Long.
IEEE Transactions on Evolutionary Computation, online 2022; volume 27(6),
1735–1749, 2023. [DOI](https://doi.org/10.1109/TEVC.2022.3227120),
[repo tác giả và manuscript](https://github.com/cuonglvsoict/EME-BI_TEVC22).
Đã đọc abstract và phần phương pháp III từ manuscript trong repo tác giả.

**Trong paper:** điều chỉnh chuyển tri thức theo chất lượng transfer; phân bổ
offspring bằng biased skill-factor inheritance; chọn trong nhiều search
operators theo hiệu quả và điều chỉnh kích thước population. Tương tác giữa
hai task không mặc nhiên có lợi theo cả hai chiều.

**Đề xuất áp dụng — ưu tiên 2:** dùng một tập operator gồm mutation một gene,
mutation cả cặp, crossover theo block, và random immigrant. Cấp thêm budget
cho operator có tiến bộ calibration Pareto tốt trên mỗi đơn vị compute,
nhưng giữ xác suất khám phá tối thiểu. Đó là adaptation từ paper, không phải
chép nguyên EME-BI; chỉ một bài toán thì chưa cần cơ chế multifactorial đầy đủ.

Nếu phát triển multitasking, nên bắt đầu với các tập prompt/timestep của cùng
UNet và cùng cách mã hóa. Mỗi candidate cuối vẫn phải qua tất cả strata.
Đừng chuyển gene trực tiếp giữa VAE của Stable Signature và UNet SleeperMark:
khác kiến trúc và ý nghĩa tọa độ. Có thể chia sẻ kinh nghiệm chọn operator,
nhưng hiệu quả transfer phải được đo riêng.

Hai mục tiêu quality/proxy không tự động là hai task. “Multitasking” và
“multi-objective” là hai cấu trúc bài toán khác nhau.

### C. HSEvo: Elevating Automatic Heuristic Design with Diversity-Driven Harmony Search and Genetic Algorithm Using LLMs

Pham Vu Tuan Dat, Long Doan, Huynh Thi Thanh Binh; AAAI 2025.
[Proceedings chính thức](https://ojs.aaai.org/index.php/AAAI/article/view/34898),
[full text](https://arxiv.org/html/2412.14995v1),
[code tác giả](https://github.com/datphamvn/HSEvo).

**Trong paper:** nghiên cứu đa dạng trong evolutionary program search với LLM;
dùng code embeddings để phân tích diversity; Harmony Search tinh chỉnh tham
số của heuristic tốt; kết hợp flash reflection. Đối tượng tiến hóa là heuristic
program, không phải model weights.

**Đề xuất áp dụng — ưu tiên 3:** cân bằng tìm kiếm rộng với tinh chỉnh cục bộ
trên elite. Dành một phần ngân sách cố định cho các bước scale/rounding lân
cận, sau đó đánh giá hard W4 thực. Mọi lần thử local search phải tính vào cùng
ngân sách; không so GA 156 lần đánh giá với hybrid 156+N lần rồi quy ưu thế cho
thuật toán. Với genome số nguyên nhỏ hiện tại, không cần đưa LLM vào vòng lặp.

### D. Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization

Minh Hieu Ha và cộng sự, có Huynh Thi Thanh Binh; MPaGE.
Preprint 2025, bản v3 tháng 01/2026 ghi accepted AAAI-26.
[Metadata](https://arxiv.org/abs/2507.20923),
[full text v3](https://arxiv.org/html/2507.20923v3).

**Trong paper:** chia objective space thành Pareto Front Grid để chọn đại diện
và hỗ trợ sinh heuristic; kết hợp semantic clustering/reflection bằng LLM.

**Đề xuất áp dụng:** lưu archive trải đều theo trade-off quality/proxy thay vì
chỉ tập trung vào một vùng. Nếu dùng hypervolume làm tín hiệu cấp budget,
chuẩn hóa và reference point phải cố định trước từ TRAIN/pilot; reference thay
đổi có thể tạo tiến bộ giả. Grid archive phải so với NSGA-II crowding đang có,
không mặc nhiên tốt hơn. Bài này đáng tham khảo về quản lý archive hơn là lấy
nguyên hệ thống LLM sinh chương trình.

### E. QDEvo: A Multi-Objective Quality-Diversity Framework for Automated Heuristic Design

Nam Do Khanh và cộng sự; Binh Huynh Thi Thanh là đồng tác giả.
arXiv tháng 07/2026; metadata ghi **poster GECCO 2026, 4 trang**, không gọi là
full paper. [Metadata](https://arxiv.org/abs/2607.11916),
[full text](https://arxiv.org/html/2607.11916v1).

**Trong paper:** archive các heuristic, clustering bằng code embedding, cạnh
tranh Pareto trong từng cluster và hierarchical reflection để giữ diversity.

**Đề xuất áp dụng — ưu tiên ngang linkage:** chromosome khác nhau chưa chắc
làm model hành xử khác nhau. Sau forward calibration vốn đã có, ghi descriptor
gồm RMS thay đổi prediction theo timestep, phân bố thay đổi theo block và tỉ
lệ thay mã W4. Chọn elite trong các nhóm hành vi khác nhau nhưng vẫn đạt quality
gate. Đây là chuyển ý tưởng diversity sang model behavior; không phải cơ chế
đo watermark. Không dùng BA/TPR hay dữ liệu owner để tạo descriptor.

Archive của chúng ta nên có giới hạn, chỉ lưu gene/descriptor/metrics; không
lưu một UNet hay activation đầy đủ cho mỗi cá thể. Exact duplicate có thể
cache fitness nếu fingerprint bao gồm cả codes và scales. Descriptor gần nhau
chỉ là xấp xỉ, không đủ để khẳng định hai model tương đương.

## 2. Thiết kế thử nghiệm tôi ưu tiên

Không ghép cả năm paper vào một phương pháp rồi chạy ngay. Bắt đầu bằng ba
thay đổi tách được đóng góp:

1. **Crossover có cấu trúc:** giữ nguyên cặp scale–bias, sau đó kiểm tra liên
   kết giữa các block. Học từ MF-LTGA, trước mắt dùng structure prior đơn giản.
2. **Archive theo hành vi calibration:** giữ những cách phân bố quantization
   error khác nhau ở cùng chất lượng; học nguyên tắc diversity từ HSEvo/QDEvo.
3. **Chọn operator thích nghi:** đo hiệu quả từng operator trên FIT, lấy cảm
   hứng từ EME-BI. Không dùng SELECT để cập nhật policy mỗi thế hệ.

Protocol: random → GA hiện tại → GA + paired crossover → GA + behavior archive
→ kết hợp; chỉ thêm adaptive operator sau khi có đủ quan sát. Random grouping
cùng kích thước là đối chứng quan trọng cho claim linkage. Có thể pilot với
156 evaluation, nhưng population nhỏ không đủ để kết luận về learned linkage.
Phải tính cả forward xây descriptor, local search và validation vào ngân sách.

Descriptor đề xuất (chưa triển khai):

`b(psi) = [RMS(delta_prediction | early), RMS(delta_prediction | middle),
RMS(delta_prediction | late), code_change_fraction_per_block]`.

Timestep chỉ là cách đo/chia calibration task; vẫn một bộ trọng số lượng tử
hóa cố định lúc inference. Nếu thay quantizer theo timestep, đó là action space
khác và phải khai báo riêng. Với Stable Signature, cần descriptor trên decoder
latent→image thay vì giả định có noise-prediction timestep như UNet.

## 3. Điều quyết định hơn lựa chọn GA

Kết quả đã đọc trước survey: CFG SleeperMark BA 98,67%, TPR 100/100, SSIM
triggered 0,732. Coherent correction rất nhỏ ở các thời điểm log. Đây chỉ ra
hai khả năng cần tách: optimizer chưa tìm tốt, hoặc proxy không tác động đến
watermark. Các paper trên chủ yếu giúp khả năng thứ nhất.

Thử nghiệm cần trả lời tuần tự:

- Ở cùng compute, search mới có cải thiện FIT và SELECT thực sự không?
- Tại cùng mức chất lượng ảnh, cải thiện proxy có đi cùng giảm TPR ở đánh giá
  độc lập không? Không chỉ chọn loss thấp hơn hoặc perturbation lớn hơn.
- Cặp gene/block có tương tác đo được hay chỉ là một heuristic grouping?
- Hiệu ứng có lặp lại với prompts/seeds và model/key mới được giữ lại không?

Các TEST run đã dùng để chỉnh hướng nghiên cứu nên được coi là development
evaluation. Kết luận cuối cần một tập cuối giữ riêng sau khi chốt phương pháp.
Owner score dùng phân tích sau freeze; không quay lại chọn candidate trên cùng
TEST rồi gọi toàn bộ quá trình là blind selection.

## 4. Novelty nên đặt ở đâu

“Áp MF-LTGA/HSEvo cho quantization” riêng lẻ chưa đủ mạnh cho bài phương pháp.
Giả thuyết đáng kiểm tra hơn: các tương tác giữa quantizer trong diffusion cho
phép tạo những sai số bù nhau về chất lượng nhưng khác nhau về ảnh hưởng tới
watermark. Linkage/behavior analysis có thể giúp khám phá và kiểm chứng điều
đó. Hiện đây chỉ là giả thuyết, không phải một phát hiện đã có bằng chứng.

Ưu tiên thực tế: **paired/block crossover và behavior diversity trước; adaptive
operator sau; multitasking xuyên hai watermark và LLM sinh thuật toán để sau**.
Các cơ chế tối ưu học được từ nhóm tác giả có ích, nhưng không cho phép cam
kết BA 0,5–0,6 hay novelty “đầu tiên”.
