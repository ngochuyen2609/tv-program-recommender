3.5 Content-based Weighted

3.5.1. Ý tưởng và phương pháp tiếp cận

Các hệ thống Content-based thuần túy thường bị giới hạn bởi việc chỉ gợi ý các sản phẩm "giống y hệt" quá khứ, thiếu tính khám phá. Ngược lại, Collaborative Filtering (CF) lại gặp khó khăn với item mới.

Chúng tôi đề xuất phương pháp Hybrid Semantic Model với cơ chế Implicit Weighted Training.

Kết hợp sức mạnh: Sử dụng mô hình ngôn ngữ Sentence-BERT để hiểu nội dung item (giải quyết Cold-start) và User Embedding để học sở thích cá nhân (giải quyết Personalization).

Cơ chế trọng số: Không coi mọi lượt xem là như nhau. Chúng tôi tính toán "điểm quan tâm" dựa trên thời lượng xem và thời gian trôi qua, từ đó ưu tiên học từ các tương tác chất lượng cao.

3.5.2. Xử lý dữ liệu

3.5.2.1. Mã hóa ngữ nghĩa

Để mô hình hiểu được nội dung chương trình, chúng tôi xây dựng văn bản đại diện Tp cho mỗi item:
Tp= CategoryGenre⊕Director⊕Channel
Văn bản này được đưa qua mô hình Sentence-BERT (paraphrase-multilingual-MiniLM-L12-v2) để trích xuất vector đặc trưng cố định vcontentℝ384. Vector này nắm bắt ngữ nghĩa đa ngôn ngữ của chương trình.

3.5.2.2. Tính toán phản hồi ẩn

Dữ liệu log thô chỉ có thông tin xem/không xem. Chúng tôi chuyển đổi sang điểm số liên tục rui để đo lường mức độ yêu thích thực sự:
rui=(screen_timeui1.2)exp(−0.02days_ago)
screen_time1.2: Thưởng cho các lượt xem có thời lượng dài (tăng tính phi tuyến).

exp(−0.02days_ago): Hàm suy giảm theo thời gian (Time Decay). Các hành vi gần đây có trọng số cao hơn hành vi trong quá khứ xa.

3.5.2.3. Lấy mẫu có trọng số

Thay vì sử dụng toàn bộ dữ liệu, chúng tôi áp dụng chiến lược lấy mẫu dựa trên rui. Các tương tác có rui cao (xem nhiều, xem gần đây) có xác suất được đưa vào tập huấn luyện cao hơn, giúp mô hình tập trung học các sở thích quan trọng nhất.

3.5.3. Kiến trúc mô hình

Mô hình là một mạng Neural lai (Hybrid Neural Network) gồm hai nhánh:

Nhánh người dùng:

Mỗi người dùng được đại diện bởi một vector học máy Euℝ128 (được khởi tạo ngẫu nhiên và cập nhật qua lan truyền ngược).

Nhánh sản phẩm:

Đầu vào là vector SBERT cố định (384 chiều).

Đi qua một mạng Adapter (Feed-forward Network) để chiếu không gian ngữ nghĩa sang không gian sở thích:

Ei= Linear2(ReLU(Linear1(vcontent)))

Kích thước đầu ra: $128$ chiều (khớp với user).

Điểm số dự đoán: Là tích vô hướng giữa hai vector đã biến đổi:
yui=EuEi.
3.5.4. Cấu hình huấn luyện & Hàm mất mát

3.5.4.1. Hàm mất mát

Sử dụng BPR Loss (Bayesian Personalized Ranking), tối ưu hóa thứ hạng cặp đôi:
ℒ=−(u,i,j)Dln(yui−yuj)
Mô hình được huấn luyện để điểm của cặp "User - Item đã xem" (i) phải lớn hơn điểm của cặp "User - Item ngẫu nhiên" (j).

3.5.4.2. Tham số thực nghiệm

Model: paraphrase-multilingual-MiniLM-L12-v2 (SBERT).

Batch Size: 4096 (Tối ưu tốc độ huấn luyện).

Hidden Dimension: 128.

Learning Rate: 0.001.

Epochs: 3 (Do dữ liệu lớn và dùng pre-trained embedding nên mô hình hội tụ rất nhanh).

Optimizer: Adam.

3.5.5. Chiến lược sinh gợi ý & Hậu xử lý

Quy trình tạo danh sách gợi ý cho giai đoạn Submission (Test phase) bao gồm bước tăng cường độ phổ biến (Popularity Boosting):

Tính điểm tương đồng: Scorebase=EuEitem

Điều chỉnh theo độ phổ biến: Để tránh việc gợi ý các item quá lạ lẫm mà user ít có khả năng click, chúng tôi cộng thêm một bias nhẹ từ độ phổ biến của item (Popularity):
Final Score = Scorebase(1+0.1ln(Popularityi))
Top-K Selection: Chọn 5 item có điểm cao nhất để gợi ý.

3.5.6. Tóm tắt

3.5.6.1. Ưu điểm so với Collaborative Filtering (CF)

Phương pháp đề xuất khắc phục được 3 điểm yếu của các thuật toán CF thuần túy (như Matrix Factorization hay User/Item-KNN):

Giải quyết triệt để vấn đề Cold-Start: Nhờ Sentence-BERT, bất kỳ chương trình mới nào ngay khi có metadata (tiêu đề, mô tả) đều được tạo ngay một vector ngữ nghĩa chất lượng cao vcontent, cho phép hệ thống gợi ý nó cho người dùng phù hợp ngay lập tức.

Khả năng hiểu ngữ nghĩa sâu) Các item có nội dung văn bản tương đồng sẽ có vector SBERT gần nhau trong không gian tiềm ẩn, giúp hệ thống gợi ý được các chương trình cùng thể loại/phong cách ngay cả khi dữ liệu tương tác còn thưa thớt.

Lọc nhiễu bằng phản hồi ẩn : Sử dụng cơ chế Implicit Weighting (dựa trên thời lượng xem và độ trễ thời gian) giúp mô hình tập trung học từ các tương tác "chất lượng cao", phản ánh đúng sở thích thực sự của người dùng.

3.5.6.2. Các hạn chế tồn tại

Bên cạnh các ưu điểm, phương pháp hiện tại vẫn tồn tại những hạn chế cần được cải thiện trong tương lai:

Phụ thuộc vào chất lượng Metadata: Hiệu quả của nhánh Content-based phụ thuộc hoàn toàn vào chất lượng văn bản đầu vào. Nếu metadata sơ sài, sai lệch hoặc thiếu thông tin, vector SBERT sinh ra sẽ không chính xác, kéo theo chất lượng gợi ý đi xuống.

Hàm tương tác đơn giản: Hiện tại mô hình sử dụng tích vô hướng để tính điểm tương thích. Hàm này tuy nhanh nhưng có thể chưa nắm bắt được các mối quan hệ phi tuyến tính phức tạp giữa User và Item như các mô hình Deep Learning sâu hơn (ví dụ: Neural Collaborative Filtering).
