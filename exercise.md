1. Chuẩn bị tài liệu nộp
Tài liệu nộp cho phần thực nghiệm bao gồm ba thành phần bắt buộc:

Báo cáo viết: mô tả kiến trúc chương trình và giải thích mã nguồn

Link dữ liệu: đường dẫn gốc tới các tập dữ liệu sử dụng trong thực nghiệm, lưu trong tập tin văn bản thuần (plain-text)

Mã nguồn chương trình:

Dạng tập tin nén hoặc link GitHub

Lưu ý: cần đảm bảo quyền truy cập cho GVLT (nnthao@fit.hcmus.edu.vn) và trợ giảng (hlhdang@fit.hcmus.edu.vn) mà vẫn bảo mật được mã nguồn.

2. Yêu cầu về hình thức báo cáo
Báo cáo soạn thảo trên Microsoft Word: Giãn dòng: 1.5, Font size: 13, và Lề: mặc định.

Nếu sử dụng công cụ khác (LaTeX, Google Docs, …) → Định dạng tương đương Word

Bắt buộc nộp file cuối cùng ở định dạng PDF

3. Yêu cầu về nội dung báo cáo
Báo cáo cần trình bày theo đúng thứ tự các mục sau:

(1) Danh sách thành viên
Mã số học viên – Họ tên

Công việc cụ thể của từng thành viên

Mức độ đóng góp (0–100%)

Ví dụ:

Hoàn thành đầy đủ → 100%

Làm một nửa → 50%

(2) Checklist mức độ hoàn thành
Lập bảng checklist thể hiện mức độ hoàn thành, ký hiệu tại mỗi mức đánh giá được quy ước như sau

✔: Hoàn thành

! : Đang thực hiện (chưa hoàn tất)

✘: Chưa thực hiện

(3) Mức 1
Mô tả tập dữ liệu:

Tên dataset

Đường dẫn gốc tới các tập dữ liệu

Thống kê: số mẫu, số thuộc tính, độ phân giải ảnh / độ dài câu, các lưu ý khác

Mô tả mô hình tiền huấn luyện:

Mô tả kiến trúc của mô hình, cấu hình huấn luyện (lượng dữ liệu huấn luyện, có xử lý dữ liệu trước hay không, thời gian huấn luyện, optimizer, cấu hình máy đã dùng, nếu thông tin nào không có thì bỏ qua)

Chỉ rõ file chứa mô hình trong tài liệu nộp

Kết quả thực nghiệm:

Bảng so sánh:

Kết quả trong bài báo

Kết quả nhóm tự chạy

Bắt buộc có phân tích và nhận xét

(4) Mức 2
Thực hiện tương tự Mức 1

(5) Mức 3
Chọn một trong hai hướng:

Chiến lược 1 (Fine-tuning):

Mô tả chi tiết cách fine-tune

Bảng kết quả thực nghiệm

Không cần mô tả lại dữ liệu

Chiến lược 2 (Ablation study):

Mô tả rõ các kịch bản ablation

Phân tích ảnh hưởng của từng thành phần

(6) Mức 4
Trình bày kiến trúc mô hình sau khi chỉnh sửa (có hình minh họa)

So sánh:

Mô hình trước vs sau chỉnh sửa

Phân tích:

Lý do và ý nghĩa của các thay đổi (về mặt lý thuyết)

Báo cáo kết quả thực nghiệm trên:

Dataset ở Mức 1

Dataset ở Mức 2