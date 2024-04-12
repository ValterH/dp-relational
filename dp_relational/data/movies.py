import pandas as pd
from dp_relational.lib.dataset import Table, RelationalDataset

file_path = "movieLens/movies.dat"
movies_df = pd.read_csv(file_path, delimiter='::', header=None, names=['ID', 'Title', 'Genres'], encoding='latin1', index_col=[0])
movies_df = movies_df.reset_index()
movies_df = movies_df.drop('Title', axis=1)

# 0-1 encode all of the movie genres:
movie_genres = [
"Action",
"Adventure",
"Animation",
"Children\'s",
"Comedy",
"Crime",
"Documentary",
"Drama",
"Fantasy",
"Film-Noir",
"Horror",
"Musical",
"Mystery",
"Romance",
"Sci-Fi",
"Thriller",
"War",
"Western"
]

for name in movie_genres:
    movies_df["is_"+name] = movies_df['Genres'].str.contains(name) * 1
    assert sum(movies_df["is_"+name]) > 0
movies_df = movies_df.drop('Genres', axis=1)

file_path = "movieLens/users.dat"
users_df = pd.read_csv(file_path, delimiter='::', header=None, names=['ID', 'Gender', "Age", "Occupation", "Zipcode"], encoding='latin1', index_col=[0])
users_df = users_df.reset_index()
users_df = users_df.drop('Zipcode', axis=1)

file_path = "movieLens/ratings.dat"
ratings_df = pd.read_csv(file_path, delimiter='::', header=None, names=["UserID", "MovieID", "Rating", "Timestamp"], encoding='latin1', index_col=[0])
ratings_df = ratings_df.reset_index()

movies_table = Table(movies_df, 'ID', do_onehot_encode=[])
users_table = Table(users_df, 'ID')

dataset = RelationalDataset(movies_table, users_table, ratings_df, 'MovieID', 'UserID', dmax=10)