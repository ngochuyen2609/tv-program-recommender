# I. LOG DATA (logs_train.parquet, logs_val.parquet)
Dữ liệu log thể hiện hành vi xem TV của người dùng.
Mỗi dòng là một lượt xem (viewing event).

| **Tên cột**         | **Mô tả**                                                        | **Mục đích sử dụng**                               |
| ------------------- | ---------------------------------------------------------------- | -------------------------------------------------- |
| **user_id**         | ID ẩn danh của người dùng                                        | Khóa dùng cho CF, phân tích hành vi, session       |
| **tv_show_id**      | ID chương trình/phim/tập; giá trị `0` là không map được metadata | Join với metadata, item ID                         |
| **vsetv_id**        | Kênh TV nơi nội dung được phát                                   | Liên kết metadata theo kênh                        |
| **start_time_view** | Thời điểm bắt đầu xem                                            | Dùng cho session segmentation, mô hình time-aware  |
| **end_time_view**   | Thời điểm kết thúc xem                                           | Dùng cho phân tích thời gian xem                   |
| **duration_view**   | Thời lượng người dùng xem (giây)                                 | Xây implicit rating, đánh giá chất lượng hành vi   |
| **screen_time**     | Tỷ lệ (0–1): thời gian xem / thời lượng chương trình             | Thước đo mức quan tâm thật → làm implicit feedback |
| **session_id**      | ID phiên xem (group theo gap < 45 phút)                          | Dùng cho sequential recommendation / co-visitation |


# II. METADATA (metadata.parquet, metadata_test.parquet)
Metadata mô tả nội dung của từng chương trình truyền hình.

| **Tên cột**          | **Mô tả**                                     | **Mục đích sử dụng**                       |
| -------------------- | --------------------------------------------- | ------------------------------------------ |
| **tv_show_id**       | ID duy nhất của chương trình (item)           | Join với log, khóa nội dung                |
| **vsetv_id**         | ID kênh phát sóng nội dung                    | Mapping theo kênh                          |
| **tv_show_category** | Thể loại chính (news, movie, sports, kids...) | Input cho content-based & hybrid           |
| **director**         | Đạo diễn hoặc MC/host; có thể trống           | Feature metadata, encode → ID/hashing      |
| **genres_list**      | Danh sách thể loại phụ (list string)          | Multi-label content features, embedding    |
| **actors_list**      | Danh sách diễn viên/MC                        | Đặc trưng nội dung quan trọng cho phim     |
| **duration**         | Thời lượng chương trình (giây)                | Chuẩn hóa screen_time, đánh giá chất lượng |
| **start_time**       | Thời điểm chương trình bắt đầu phát           | Dùng để join theo khoảng thời gian với log |
| **prog_end**         | Thời điểm chương trình kết thúc               | Tính overlap khi join logs ↔ metadata      |


# III. Mục đích sử dụng dữ liệu
1. Logs dùng để:
- Huấn luyện mô hình Collaborative Filtering
- Xây dựng implicit rating từ screen_time
- Xây session → sequential modeling
- Phân tích cold-start (train vs val vs test)

2. Metadata dùng để:
- Bổ sung thông tin content cho Hybrid models
- Trích xuất embedding (text, genres, actors)
- Giảm cold-start item
- Tạo item-feature matrix cho DeepFM / LightFM

# IV/
## 1. Quy mô & độ thưa (Size & Sparsity)
Số user: 4,838
Số item (tv_show_id ≠ 0): 3,716
Số tương tác: 931,775
Mật độ ma trận user–item (density): ~0.0518 (≈ 5.18%)
➡ Matrix rất thưa → phù hợp cho các mô hình CF, embedding, candidate generation top-K.

## 2. Phân bố theo user và item
Per User
p10: user xem ≥ 17 chương trình
median: 66.5
mean: 82.23
97.89% user xem ≥ 5 items
95.23% user xem ≥ 10 items
➡ User có lịch sử khá phong phú → CF học tốt.

Per Item
p10: mỗi item có ≥ 5 lượt xem
median: 44
mean: 250.75
70.4% item có ≥ 20 lượt xem
➡ Catalog tương đối “ấm”, nhiều item đủ dữ liệu học.

## 3. K-core (5 user × 5 item)
users còn lại: 4735
items còn lại: 3362
interactions còn lại: 930,553
➡ Dữ liệu cực kỳ vững → CF / Matrix Factorization hoạt động tốt.

## 4. Session analysis
median session length: 1 lượt xem
mean session length: 2.06
44.83% session có ≥ 2 lượt xem
➡ Mức độ phiên đủ tốt để dùng Co-visitation / Sequential Recommendation.

## 5. Chất lượng tín hiệu (screen_time)
p25 = 0.503
median = 0.832
p75 = 1.0
52.22% lượt xem có screen_time ≥ 0.8
➡ Tín hiệu rất sạch và mạnh → thích hợp làm implicit rating.

## 6. Cold-start (train → val)
Cold users in val: 0.61%
Cold items in val: 23.35%
➡ Gần như không có cold-user — nhưng cold-item khá cao → cần metadata / content-based để khắc phục.

## 7. Popularity Skew
Top 50 item chiếm 54.81% tổng tương tác
Top 100 item chiếm 63.57%
Top 500 chiếm 80.32%
Top 1000 chiếm 89.53%
➡ Dữ liệu lệch mạnh theo item phổ biến → cần attention khi train để tránh mô hình chỉ recommend item hot.