# Mô tả bộ dữ liệu

- Các tệp
    - dataset11-30week.csv – tập huấn luyện. vsetv_id giống với channel_id trong các tệp export_arh_*.csv.
    - export_arh_10-19.csv – thông tin bổ sung cho train (lịch phát TV). Chứa tv_show_id – bạn phải dự đoán 5 ID này cho các user_id trong submission.csv. Lưu ý: tv_show_id = 0 là chương trình không liên quan, loại bỏ!
    - export_arh_20-30.csv – thông tin bổ sung cho train (lịch phát TV). Cùng quy tắc như trên; loại tv_show_id = 0.
    - export_arh_31-42.csv – thông tin bổ sung cho test (lịch phát TV). Chứa tv_show_id – bạn phải dự đoán 5 ID này cho các user_id trong submission.csv. Loại tv_show_id = 0.
    - submission.csv – tệp mẫu ở định dạng đúng. Hãy thay 5 số 0 bằng 5 tv_show_id dự đoán cho mỗi người dùng.

- Trường dữ liệu trong Train
    - user_id – ID ẩn danh, duy nhất cho mỗi người dùng.
    - vsetv_id – ID kênh (bằng với channel_id trong export_arh_*.csv).
    - start_time – thời điểm bắt đầu xem kênh.
    - end_time – thời điểm kết thúc xem kênh.
    - duration – thời lượng xem kênh.
- Trường dữ liệu trong thông tin bổ sung (EPG)
    - channel_id – ID kênh.
    - channel_title – tên kênh.
    - start_time – thời điểm chương trình bắt đầu.
    - duration – thời lượng (giây).
    - tv_show_title – tên chương trình/TV show.
    - tv_show_id – định danh chương trình/phim trong cơ sở dữ liệu. Không liên quan đến cơ sở phim; int là tập của TV show, 0 là phim (và được coi là không liên quan).
    - tv_show_category – danh mục chương trình.
    - tv_show_genre_1/2/3 – thể loại 1/2/3.
    - year_of_production – giai đoạn phát hành hoặc năm sản xuất; nếu kết thúc bằng “-” nghĩa là đến hiện tại.
    - director – đạo diễn.
    - actors – diễn viên/MC.

- Dữ liệu test chứa sở thích chương trình TV của người dùng trong 12 tuần tiếp theo (31–42). Public/private test tách theo người dùng (50/50), không theo thời gian. Mỗi người dùng trong lời giải có đúng 5 chương trình ưa thích, sắp tăng dần từ ưu tiên nhất (#1 – có tổng thời gian xem lớn nhất) đến #5 (ít nhất trong top 5) đối với từng người dùng.

- Cách tạo “tệp đáp án chuẩn” (logic ẩn sau đánh giá)
    - Loại tất cả bản ghi có tv_show_id = 0 khỏi các tệp lịch phát.
    - Với mỗi người dùng và mỗi chương trình, tính screen time = (thời gian người dùng xem chương trình) / (tổng thời lượng chương trình). Chỉ giữ các chương trình có screen time ≥ 0.8.
    - Với mỗi người dùng, tính tần suất xuất hiện của các chương trình có screen time cao.
    - Với mỗi người dùng, chọn đúng 5 chương trình có tần suất cao nhất, sắp xếp giảm dần ( #1 cao nhất → #5 thấp nhất trong top 5 ). Thứ tự có ý nghĩa!

- Nhiệm vụ của bạn: dự đoán 5 tv_show_id (cách nhau bằng dấu cách) cho mỗi user – điền vào submission.csv.

# Preprocessing
1. Đọc & làm sạch log xem kênh
- Chuẩn hóa cột thời gian, đổi tên cột cho thống nhất, lọc lượt xem quá ngắn. (trong load_raw.load_logs)

2. Đọc & chuẩn hóa lịch phát sóng (metadata)
- Gộp các file export_arh_*, parse thời gian, tính prog_end = start_time + duration, và (rất quan trọng) lọc tv_show_id != 0 ở đây. (trong load_raw.load_meta, ta thêm một dòng filter)

3. Ghép log ↔ lịch phát sóng theo kênh + khoảng thời gian (interval join)
- Tìm chương trình chồng lấn nhiều nhất với mỗi lượt xem, tính overlap (s). (trong join_program._pick_overlap)
- Cần giữ lại show_duration_s và/hoặc tự tính từ prog_end - start_time để sau đó tính screen_time = overlap / show_duration_s.

4. Lọc theo screen_time
- Sau khi có overlap và show_duration, giữ lại các bản ghi có screen_time ≥ 0.8.

5. Tổng hợp tần suất theo user → chọn đúng 5 tv_show_id phổ biến nhất (order DESC)
- Dùng tần suất (và quy tắc phá hòa nếu cần) để tạo 5 ID cho mỗi user, đúng thứ tự.