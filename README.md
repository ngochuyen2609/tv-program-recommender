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

