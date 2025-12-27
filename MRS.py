import pandas as pd
from surprise import Dataset, Reader
from surprise import SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load the dataset
# Adjust this to point to where your .dat files are stored

# MovieLens dataset
# Load ratings data
ratings = pd.read_csv('Movie Recommendation\\ml-1m\\ratings.dat', sep='::', header=None, engine='python', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    
# Load movies data
movies = pd.read_csv('Movie Recommendation\\ml-1m\\movies.dat', sep='::', header=None, engine='python', names=['movie_id', 'movie_name', 'genre'], encoding='ISO-8859-1')

# Prepare the data for Surprise
reader = Reader(rating_scale=(1, 5))  # ratings range from 1 to 5

# Convert the ratings dataframe into a Surprise dataset
data = Dataset.load_from_df(ratings[['user_id', 'movie_id', 'rating']], reader)

# Split the data into training and test sets
trainset, testset = train_test_split(data, test_size=0.25)

# Use the SVD (Singular Value Decomposition) algorithm to predict ratings
model = SVD()

# Train the model
model.fit(trainset)

# Predict ratings on the test set
predictions = model.test(testset)

# Calculate and print the RMSE (Root Mean Squared Error)
rmse = accuracy.rmse(predictions)
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Making a prediction for a specific user and movie
user_id = str(1)  # user ID (as string)
movie_id = str(50)  # movie ID (as string)

# Predict the rating for a specific user and movie
predicted_rating = model.predict(user_id, movie_id)
print(f'Predicted rating for User {user_id} for Movie {movie_id}: {predicted_rating.est}')

# Recommending top-N movies for a specific user (e.g., User 1)
def get_top_n(predictions, n=10):
    '''Returns the top-N recommended items for each user from a list of predictions.'''
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))

    # Now sort the predictions for each user and get the top-N
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# Get top-10 movie recommendations for each user
top_n = get_top_n(predictions, n=10)

# Display the top-10 movie recommendations for User 1
print(f'Top-10 recommended movies for User 1:')
for user_id, movie_ratings in top_n.items():
    # movie_ratings is a list of tuples (movie_id, rating)
    for movie_id, rating in movie_ratings:
        print(f"User {user_id} rated Movie {movie_id} with a rating of {rating}")
