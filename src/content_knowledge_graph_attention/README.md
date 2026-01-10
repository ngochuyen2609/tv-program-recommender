3.4 Content-based + Knowledge Graph Attention

3.4.1. Ý tưởng

Phương pháp được vận hành dựa trên 3 trụ cột kỹ thuật chính:
Mã hóa ngữ nghĩa:
Sử dụng mô hình ngôn ngữ tiền huấn luyện Sentence-BERT để đọc hiểu metadata (Tiêu đề, Thể loại, Diễn viên...).
Biến đổi thông tin văn bản này thành một vector đặc trưng khởi tạo cho mỗi chương trình. Nhờ bước này, ngay cả một bộ phim mới tinh chưa có lượt xem nào cũng đã có một vị trí chính xác trong không gian vector (nằm cạnh các phim có nội dung tương tự).
Xây dựng đồ thị tri thức toàn cục (Global Graph Construction):
Hợp nhất dữ liệu từ quá khứ (Train) và hiện tại (Validation) để tạo nên một đồ thị kết nối liền mạch.
Đồ thị này đóng vai trò là "bản đồ dẫn đường", giúp liên kết các chương trình với nhau thông qua các nút trung gian.
Lan truyền sở thích qua cơ chế chú ý (Attentive Propagation):
Sử dụng mạng KGAT để lan truyền thông tin trên đồ thị.
Cơ chế Attention giúp mô hình tự động đánh giá mức độ quan trọng

3.4.2. Xử lý dữ liệu và xây dựng đồ thị

3.4.2.1. Mã hóa ngữ nghĩa

Để máy tính "hiểu" được nội dung chương trình, chúng tôi thực hiện quy trình mã hóa văn bản:

Bước 1: Tổng hợp văn bản Tp: Với mỗi chương trình p, tạo một văn bản đại diện bằng cách ghép nối các trường metadata quan trọng:

Tp = [Title] [Genre] [Actors] [Director]

Bước 2: Vector hóa: Sử dụng mô hình tiền huấn luyện Sentence-BERT (all-MiniLM-L6-v2) để chuyển đổi Tp thành vector xsem ℝ384.

Lý do chọn SBERT: Mô hình này được tối ưu hóa cho bài toán Semantic Similarity, giúp hai chương trình có mô tả giống nhau sẽ có cosine similarity cao ngay từ đầu.

3.4.2.2. Định nghĩa đồ thị tri thức

Xây dựng đồ thị không đồng nhất G = (V, E) với cấu trúc chi tiết:

Tập Đỉnh (V): Gồm 3 loại nút chính:

User Nodes (u): Đại diện người dùng.
Item Nodes (i): Đại diện chương trình TV.
Entity Nodes (e): Các thuộc tính (Actor, Director, Genre, Channel).

Tập Cạnh (E):

Interaction Edges: u⟶i (Trọng số cạnh có thể là screen_time hoặc 1).
Knowledge Edges: i⟶eactor, i ⟶egenre v.v.

3.4.3. Kiến trúc mô hình

Kiến trúc mô hình bao gồm 3 tầng xử lý tuần tự:

3.4.3.1. Tầng Khởi tạo Đặc trưng

Đối với nút Item(hi(0)): Chúng tôi sử dụng vector ngữ nghĩa xsem từ SBERT, chiếu qua một lớp Tuyến tính (Linear Projection) để giảm chiều dữ liệu và đưa về không gian ẩn của đồ thị:

hi(0) = Wprojxsem +bproj

Trong đó: Wproj ℝdmodel384, dmodel = 128

Đối với nút User và Entity (hu(0), he(o)): Do không có mô tả văn bản, các nút này được khởi tạo bằng Embedding Lookup Table với các giá trị ngẫu nhiên tuân theo phân phối Xavier Uniform

3.4.3.2. Tầng lan truyền chú ý

Sử dụng kiến trúc GATConv (Graph Attention Network Convolution) để cập nhật vector đại diện. Quá trình này mô phỏng việc "người dùng bị ảnh hưởng bởi nội dung họ xem" và "chương trình được định nghĩa bởi thuộc tính của nó".
Tại lớp l, vector của nút i được cập nhật từ các láng giềng Ni như sau:

Tính hệ số chú ý: Mức độ quan trọng eij của láng giềng j đối với nút i:
eij = LeakyReLU(aT[Whi(l)||Whj(l)])
Sau đó chuẩn hóa bằng hàm Softmax để có trọng số ij:
ij= exp(eij)kNiexp(eik)

Tổng hợp thông tin:
hi(l+1)= σ(jNiijWhjl)
Trong đó: là hàm kích hoạt (ELU hoặc ReLU). Chúng tôi sử dụng Multi-head Attention (với K = 4 heads) để ổn định quá trình học, kết quả đầu ra là trung bình cộng của các heads.

3.4.3.3. Tầng Dự đoán

Sau L lớp lan truyền (thường L =2), ta thu được vector cuối cùng hu và hi. Điểm tương thích giữa User u và Item i được tính bằng tích vô hướng:
yui=(hu)Thi

3.4.4. Cấu hình huấn luyện & Hàm mất mát

3.4.4.1. Hàm mất mát (Loss Function)

Bài toán được định nghĩa là dự đoán liên kết: xác định xác suất tồn tại cạnh kết nối giữa hai thực thể. Chúng tôi sử dụng hàm mất mát Binary Cross Entropy with Logits (nn.BCEWithLogitsLoss), kết hợp lớp Sigmoid và phép tính Entropy nhị phân để đảm bảo tính ổn định số học.
Công thức tổng quát:
ℒ = −1|ℰ|(u,i)ℰ[yuilog((sui))+(1−yui)log(1−(sui))]
Positive Sample (yui=1): Các cạnh thực tế tồn tại trong tập dữ liệu huấn luyện.

Negative Sample (yui = 0): Các cạnh giả được sinh ngẫu nhiên trong mỗi epoch với tỷ lệ 1:1 so với mẫu dương.

3.4.4.2. Tham số thực nghiệm

Embedding Dimension (d): 128 (Kích thước không gian ẩn, phù hợp để chứa thông tin từ SBERT nén xuống).

Số lượng Epochs: 300 (Huấn luyện sâu để đảm bảo sự hội tụ trên đồ thị toàn cục).

Tốc độ học (Learning Rate): 0.001 (Learning rate tiêu chuẩn cho Adam optimizer).

Optimizer: Adam (Sử dụng cấu hình mặc định, không áp dụng Weight Decay bổ sung).

Chiến lược Batch: Full-Batch Training (Toàn bộ các cạnh trong tập huấn luyện được đưa vào mô hình trong mỗi epoch để tính toán gradient).

Kiến trúc mạng: 2 lớp GATConv, 4 Attention Heads, Mean Aggregation.

3.4.5. Quy trình sinh gợi ý

Sau khi huấn luyện, chúng tôi không dùng mô hình để dự đoán điểm cho từng cặp . Thay vào đó, chúng tôi sử dụng chiến lược Approximate Nearest Neighbors (ANN) trên không gian vector:

Trích xuất Embedding: Lấy ma trận vector HitemsℝM 128 của tất cả chương trình sau khi đã học xong.

Xác định ngữ cảnh (Context Embedding): Hoặc chiến lược Item-to-Item (đơn giản hơn): Lấy vector hitem của chương trình cuối cùng user vừa xem plast.

Tìm kiếm tương đồng (Similarity Search): Tính Cosine Similarity giữa vector ngữ cảnh và toàn bộ Hitem:
Sim(plast,pj) = hplasthpj||hplast|| ||hpj||

Hậu xử lý (Post-processing):

Loại bỏ các item user đã xem trong quá khứ.
Lấy Top-K (K = 5) item có điểm cao nhất.

3.4.6. Tổng kết và đánh giá

Phương pháp Content-Enhanced KGAT mang lại sự cải thiện về mặt chất lượng biểu diễn:

Giải quyết Cold-Start: Một bộ phim mới thêm vào hệ thống vẫn có vector hi(0) rất "xịn" nhờ SBERT. Nó sẽ ngay lập tức nằm gần các phim có nội dung tương tự trong không gian vector, cho phép hệ thống gợi ý nó cho đúng đối tượng khán giả.

Khả năng giải thích: Thông qua trọng số Attention (ij), ta có thể giải thích được vì sao hệ thống gợi ý phim này

Tính ổn định: Việc kết hợp thông tin cấu trúc giúp mô hình không bị phụ thuộc hoàn toàn vào văn bản (tránh việc gợi ý sai khi mô tả phim viết không chuẩn), vì nó còn được kiểm chứng bởi hành vi cộng đồng trên đồ thị.
