## **data/interim/logs_program.parquet**
- Chứa : bảng tương tác đã JOIN đúng chương trình (level tv_show).
- Cột chính:
user_id, tv_show_id, vsetv_id, start_time_view, end_time_view, duration_view.
- Dùng để: nguồn gốc cho mọi bước sau (tạo session, train candidate, ranker).



## **data/interim/train.parquet**
- Chứa : subset của logs_program cho weeks 11–24 (không rò rỉ thời gian).
- Dùng để: train các mô hình candidate (TopPop, KNN, ALS/BPR/LightGCN, SASRec…) và ranker.
- Cột: như logs_program + session_id (sau when sessionize).


## **data/interim/val.parquet**
- Chứa : subset cho weeks 25–30.
- Dùng để: đánh giá offline (MAP@5, Recall@K, NDCG@K, ILD/Coverage) và tuning hyperparameters.

