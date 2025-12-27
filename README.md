Movie Recommendation System using SVD

A collaborative filtering–based movie recommendation system built using Singular Value Decomposition (SVD) with the Surprise library. The system predicts user–movie ratings and recommends top-N movies based on learned latent factors.

📌 Project Overview

Recommendation systems are widely used by platforms like Netflix and Amazon to personalize content.
This project implements a matrix factorization approach to:

Predict movie ratings for users

Evaluate model performance using RMSE

Generate personalized movie recommendations

🧠 Algorithm Used
Singular Value Decomposition (SVD)

Factorizes the user–item interaction matrix

Learns latent features for users and movies

Helps overcome sparsity in rating data

Library Used: Surprise

📂 Dataset

MovieLens 1M Dataset

1,000,000 ratings

~6,000 users

~4,000 movies

Dataset Source:
🔗 https://grouplens.org/datasets/movielens/1m/

⚠️ The dataset is not included in this repository due to size constraints.
Please download it manually and place it as described below.

📁 Expected Dataset Structure
Movie Recommendation/
├── ml-1m/
│   ├── ratings.dat
│   ├── movies.dat
│   └── users.dat

⚙️ Features Implemented

Data loading using Pandas

Rating scale normalization

Train-test split (75/25)

Rating prediction using SVD

RMSE evaluation

Top-N movie recommendations per user

🚀 How to Run the Project
1️⃣ Install Dependencies
pip install pandas scikit-surprise

2️⃣ Clone the Repository
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system

3️⃣ Download Dataset

Download MovieLens 1M dataset from:
https://grouplens.org/datasets/movielens/1m/

Extract into the folder structure shown above

4️⃣ Run the Script
python recommend.py

📊 Results

Metric Used: Root Mean Squared Error (RMSE)

Performance: Consistent prediction accuracy for unseen user–movie pairs

Scalability: Efficient for medium-scale recommendation tasks

🔮 Future Enhancements

Add implicit feedback support

Implement hybrid recommender (content + CF)

Hyperparameter tuning

Evaluate using Precision@K / Recall@K

Build an interactive recommendation UI

🛠️ Tech Stack

Python

Pandas

Surprise

SVD (Matrix Factorization)

👤 Author

Hanan Bhat

LinkedIn: https://www.linkedin.com/in/hanan-bhat-49a8a1269

GitHub: https://github.com/bugfikser/

📜 License

This project is intended for educational and research purposes.
