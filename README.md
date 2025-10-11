# Guideline to dowload data
## Step 1: Set up virtal environment and necessary library (Cmd )
    >> python -m venv .venv  
    >> ..venv\Scripts\activate.bat  
    >> ..venv\Scripts\python.exe -m pip install --upgrade pip  
    >> ..venv\Scripts\python.exe -m pip install -r requirements.txt  

## Step 2: Create API token for Kaggle
    - Đăng nhập Kaggle → Account → API → Create New Token → tải file kaggle.json.
    - Chép vào: C:\Users\<USERNAME>\.kaggle\kaggle.json (tạo thư mục .kaggle nếu chưa có).

## Step 3: Tham gia (Join) competition trước khi tải and accept rules
    - https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data

## Step 4: Tải dữ liệu bằng CLI
    - .\.venv\Scripts\kaggle.exe competitions download -c sweet-tv-channel -p .\data    

# Pipeline (Concise)
1) Chuẩn hoá & JOIN  
Mục tiêu: tạo tương tác user × tv_show (implicit) khớp lịch phát.  
Steps:
dataset11-30.csv → rename stop_time→end_time, duraton→duration; parse datetimes; filter duration ≥ 180s.  
JOIN with export_arh_11-20-final.csv + export_arh_21-30-final.csv:  
Key: vsetv_id (logs) ↔ channel_id (meta).  
Interval join: [user_start, user_end] ∩ [prog_start, prog_end], where prog_end = start_time + duration.  
If multiple matches → pick the largest time overlap.  
Save → data/interim/logs_program.parquet  
Columns: user_id, tv_show_id, vsetv_id, start_time_view, end_time_view, duration_view.  
Checkpoint: post-join coverage > 70% (after 180s filter).  

2) Session hoá & Split thời gian  
Session: per-user sort by time; cut when gap > 45m → session_id.  
Split (no leakage): Train weeks 11–24, Val 25–30 (Test 31–42 only for submission).  
Output: data/interim/train.parquet, data/interim/val.parquet.  

3) Candidate (multi-source)

    - CF cơ bản/nâng cao  
    TopPop (time-decay): ∑exp(−Δt/α)·log1p(duration), α=7–30d.  
    ItemKNN (cosine+shrinkage): K=200, λ=50–200.  
    ALS/WMF (implicit): factors=64, reg=0.05, iters=15, α=0.01–0.1.  
    BPR-MF: factors=128, lr=1e-3, reg=1e-4, neg=5.  
    LightGCN: layers=2–3, emb=64–128, lr=1e-3, reg=1e-4.  

    - Session-based
    Markov (order 1/2) with small smoothing (k≈1e-3).  
    SASRec: max_len=100, d=128/256, layers=2, heads=2/4, dropout=0.2.  

    - Content-based  
    TF-IDF/BM25: (title + actors + director + genres + category); TF-IDF(1–2, min_df=5) / BM25(k1=1.2, b=0.75).  
    SBERT + Faiss: all-MiniLM-L6-v2 → 128d (reduced), L2-norm, ANN cosine.  

    - Knowledge-based  
    KG encoder (GCN/GAT item-side): graph (program–genre/actor/director/channel) → item_KG_emb 64–128d.

    - Deep Learning (candidate)  
    Two-Tower (YouTube/DSSM): user/item encoders → ANN retrieval (Faiss).  
    User features: user_id_emb, pooled history, time; Item: tv_show_id_emb, channel_id_emb, SBERT_128, (opt) KG_item_64, genres.  
    MultVAE/MultDAE: reconstruct implicit user vector → top-K logits.  
    Caser/NextItNet: conv/residual next-item sequence models.  
    CL4SRec/CoSeRec: contrastive pretrain for session embeddings.  
    Item2Vec/DeepWalk/Node2Vec: cheap item embeddings for nearest-neighbors.  
    Pooling: take top-100 from each source → merge & dedup → pool ≤ 300–500 per user → data/interim/candidates.parquet.  
    Checkpoint: pool Recall@100 (val) ≥ 0.8 after full sources.  

4) Features for Ranker  
ID: user_id, tv_show_id, channel_id.  
Time: hour_sin/cos, day_of_week, recency, near_airtime.  
Content: multi-hot genres/category, SBERT_128.  
Knowledge: KG_item_64 (if available).  
CF/Seq scores: normalized [0,1] for each score_* (TopPop/KNN/ALS/BPR/LightGCN/Markov/SASRec/TFIDF/SBERT).  

5) Rankers (fusion)  
LightGBM Ranker: objective=lambdarank, lr=0.05, num_leaves=255, n_estimators=2000, group=user_id.  
DeepFM (PyTorch): fields = ID + categorical + dense (time, SBERT, KG, scores); MLP [256,128,64], dropout=0.2, Adam=1e-3.  
Labels: positives from val; negatives 5–10×/user.  
Checkpoint: MAP@5 (val) +10–30% over best single-source.  

6) Re-ranking  
MMR: λ=0.6–0.8, Sim = cosine(SBERT_128) + Jaccard(genres)/overlap(actors,director); soft quota per-genre ≤ 3/5.  
Trend/Freshness boost: boost = β·exp(−Δt/τ) with β=0.05–0.2, τ=1–7d; apply only on test (use meta weeks 31–42).  
Checkpoint: ILD/Coverage ↑, MAP@5 no drop (ideally ↑).  

7) Evaluation & Ablation  
Metrics: MAP@5, Recall@5, NDCG@5, ILD, Coverage (on val weeks 25–30).  
Ablation: −content / −session / −CF scores / −MMR / −trend; log configs & results under runs/.  

8) Final Train & Submit  

## Nội dung chính từng khối
### (a) CF cơ bản / nâng cao
- `candidates/toppop.py` — TopPop time-decay (cửa sổ 7–30 ngày).
- `candidates/itemknn.py` — ItemKNN (cosine + shrinkage).
- `candidates/als_wmf.py` — Implicit ALS/WMF (confidence từ `duration`/`count`).
- `candidates/bpr.py` — BPR-MF (pairwise ranking).
- `candidates/lightgcn.py` — LightGCN (đồ thị user–item, BPR loss).

### (b) Session-based
- `candidates/markov.py` — Markov order-1/2 chuyển kênh/chương trình theo phiên.
- `candidates/sasrec.py` — SASRec (Transformer) dự đoán next-item; xuất top-N theo từng session/user.

### (c) Content-based
- `preprocess/build_text.py` — Sinh TF-IDF/BM25 từ: title/actors/director/genres/category.
- `candidates/content_tfidf.py` — Lấy lân cận bằng cosine trên TF-IDF/BM25 (sparse).
- `candidates/content_sbert.py` — SBERT embedding + Faiss (ANN cosine) để kéo ứng viên “na ná”.

### (d) Knowledge-based
- `preprocess/build_kg.py` — Dựng đồ thị: program–genre/actor/director/channel (edges + id map).
- `kg/gcn_item_encoder.py` — GCN/GAT item-side tạo `item_KG_emb` (64–128d) dùng làm feature/candidate.

### (e) Deep Learning (candidate)
- `candidates/two_tower.py` — Two-Tower (YouTube/DSSM): encoder user/item → ANN/Faiss retrieval (top-K).
- `candidates/multvae.py` — MultVAE/MultDAE: autoencoder trên vector implicit user → lấy top-K logits.
- `candidates/caser_nextitnet.py` — Caser/NextItNet (CNN/Res) cho next-item từ chuỗi; xuất top-K.
- `candidates/cl4srec.py` — Contrastive session (CL4SRec/CoSeRec) → phiên embedding robust → ANN truy hồi.
- `candidates/item2vec.py` — Item2Vec/DeepWalk/Node2Vec: embedding dựa chuỗi/đồ thị → lân cận theo cosine.

### (f) Rankers (hợp nhất đa nguồn)
- `rankers/deepfm.py` — DeepFM: fields = user_id, item_id, genres, channel, actor_ids(top-K pooled), director, `text_emb`, `KG_emb`, time_feats, và **scores** từ candidate (ALS/KNN/LightGCN/SASRec/Content…).
- `rankers/neumf.py` — NeuMF (GMF + MLP).
- `rankers/lgbm_ranker.py` — LightGBM Ranker (lambdarank) với đặc trưng thủ công + scores.

### (g) Re-ranking
- `rerank/mmr.py` — MMR đa dạng hoá theo similarity nội dung (cosine SBERT, Jaccard genres, overlap actors/director).
- `rerank/trend.py` — Trend/Freshness boost cho item mới/sắp phát (dựa `start_time` từ metadata tuần 31–42, chỉ áp ở test).

## Strucure project
recsys_sweettv/  
  data/  
    raw/          # original CSVs  
    interim/      # joined/sessions/candidates  
    features/     # text/KG/pop/time features, embeddings  
    runs/         # logs, evals, submissions  
  configs/  
    data.yaml  
    candidate/    # toppop.yaml, itemknn.yaml, als_wmf.yaml, bpr.yaml, lightgcn.yaml,  
                  # markov.yaml, sasrec.yaml, content_tfidf.yaml, content_sbert.yaml,  
                  # two_tower.yaml, multvae.yaml  
    ranker/       # lgbm.yaml, deepfm.yaml, neumf.yaml  
    rerank/       # mmr.yaml, trend.yaml  
  src/
    utils/ # io.py, time.py, idmap.py, split.py, eval.py, logging.py  
    preprocess/   # load_raw.py, join_program.py, sessionize.py, build_text.py, build_kg.py, build_features.py  
    candidates/   # toppop.py, itemknn.py, als_wmf.py, bpr.py, lightgcn.py,  
                  # markov.py, sasrec.py, content_tfidf.py, content_sbert.py,  
                  # two_tower.py, multvae.py, pool_merge.py  
    rankers/      # lgbm_ranker.py, deepfm.py, neumf.py  
    kg/           # gcn_item_encoder.py  
    rerank/       # mmr.py, trend.py  
    pipelines/    # 00_make_data.py, 10_make_candidates.py, 20_train_ranker.py,  
                  # 30_predict_rank.py, 40_rerank_and_submit.py, 50_eval_offline.py  
  requirements.txt  
  README.md  
 
 