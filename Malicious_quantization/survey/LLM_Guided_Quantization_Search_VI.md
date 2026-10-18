# LLM hướng dẫn tìm quantizer W4 cho SleeperMark

Ngày khảo sát và triển khai: 27/09/2026.

## Kết luận thiết kế

LLM phù hợp nhất ở vai trò **đề xuất chromosome**, không làm evaluator và
không thay proxy. Mỗi thế hệ, LLM đọc một phần archive FIT gồm gene và hai
loss calibration, rồi đề xuất tối đa ba vector scale/rounding mới. Mọi vector
vẫn phải được đo bằng UNet thật. Owner trigger, extractor, key, SELECT và TEST
không xuất hiện trong prompt hay phản hồi cho LLM.

Thiết kế này lấy cảm hứng từ:

- [FunSearch, Nature 2024](https://www.nature.com/articles/s41586-023-06924-6):
  LLM sinh ứng viên nhưng evaluator có thể chạy được mới quyết định fitness.
- [ReEvo, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/4ced59d480e07d290b6f29fc8798f195-Paper-Conference.pdf):
  dùng phản hồi có cấu trúc để cải tiến heuristic qua nhiều vòng.
- [HSEvo, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34898):
  kết hợp LLM với evolutionary search và theo dõi diversity; paper tìm kiếm
  chương trình heuristic, không phải trọng số quantized.
- [MPaGE, AAAI 2026](https://arxiv.org/html/2507.20923v3): dùng Pareto-grid
  để cung cấp ngữ cảnh đa mục tiêu cho LLM. Code hiện dùng Pareto ranking đã
  có, chưa tuyên bố triển khai toàn bộ MPaGE.

Không có paper trên chứng minh LLM có thể suy ra trigger SleeperMark hoặc phá
watermark bằng quantization. Đây là một ablation về **search efficiency**.

## Hai nhánh mới

`llm_genetic_w4` dùng `Qwen/Qwen2.5-3B-Instruct`, pin revision
`aa8e72537993ba99e69dfaafa59ed015b17504d1`. Model chạy local bằng BF16 nếu
CUDA hỗ trợ, greedy decoding, không cần API key.

`local_proposal_genetic_w4` lấy Pareto elite rồi đổi một tọa độ ±1. Đây là
đối chứng bắt buộc: nếu local search bằng hoặc hơn LLM thì không thể quy lợi
ích cho khả năng suy luận ngôn ngữ.

Cả hai dùng cùng initial population, objective, FIT/SELECT split, quality
gate, population 12 và 12 thế hệ với adaptive GA. Ba proposal thay thế ba
offspring của mỗi thế hệ; tổng vẫn là 156 candidate và cùng số lần đánh giá
UNet. Chi phí LLM là compute bổ sung, được ghi bằng token, thời gian và peak
VRAM; vì vậy chỉ có candidate budget bằng nhau, không phải tổng FLOPs bằng nhau.

Prompt chỉ chứa whitelist:

- tên group;
- gene và hai fitness trên FIT;
- generation, bounds, thứ tự gene và thứ tự objective.

Đầu ra phải là đúng object JSON có duy nhất field `genes`, mỗi gene có đúng
số chiều và chỉ chứa integer trong `[-3,3]`. Output không được `exec`, `eval`
hoặc import. Proposal trùng bị bỏ. JSON lỗi tạo 0 proposal cho vòng đó và GA
thường lấp đủ population; trace ghi lỗi thay vì âm thầm gọi đó là LLM success.

Một lượt reflection tối đa được phép nếu lượt đầu lỗi hoặc không đủ gene mới.
LLM nhận danh sách gene vừa bị từ chối và số lượng còn thiếu. Tối đa ba gene
mới được chuyển tới evaluator; lượt reflection chỉ thêm inference LLM, không
thêm candidate/UNet forward. Trace giữ riêng cả hai lần sinh và token tương ứng.

## Chạy thí nghiệm

```bash
bash run_blind_quantization.sh --watermark sleepermark \
  --methods adaptive_genetic_w4 local_proposal_genetic_w4 llm_genetic_w4
```

Kết quả cần đọc:

- `*_search.json`: selected candidate, FIT/SELECT, operator và proposal trace;
- `llm_genetic_w4_llm_trace.jsonl`: prompt, response, token, parse error và thời gian;
- `parallel_summary.csv`: BA/TPR, SSIM, FID, wall time và peak VRAM.

So sánh theo thứ tự: adaptive GA với local proposal; local với LLM; sau đó mới
xem owner metric đã freeze. Nếu LLM chỉ giảm proxy nhưng TPR vẫn 100%, kết luận
là search tốt hơn trên proxy chưa đúng, không phải attack tốt hơn.

Đây là development evaluation. Việc đọc owner TEST rồi sửa prompt khiến các
run sau không còn là final held-out; paper cần prompt/seed/checkpoint hoặc key
mới sau khi chốt thiết kế.
