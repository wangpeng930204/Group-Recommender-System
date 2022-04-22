import pickle
import time

from group import generate_group, aggregate_group_rating
from measures import predictions
from processing import preprocessing, remove_missing_film, get_user_rating_dicts, get_movies_aspect_matrix, \
    compute_similarity

MUR = 0.1
MUG = 0.6
MUA = 0.1
MUD = 0.1

if __name__ == "__main__":
    movielens_data = "100k"
    # read rating data
    ratings = pickle.load(open("data/100k_benchmark_ratings.pkl", "rb"))
    # read films data
    films = pickle.load(open("data/100k_benchmark_films_movielens.pkl", "rb"))
    # remove from ratings the missing films (that were missing info and hence were discarded)
    ratings, films = remove_missing_film(films, ratings)
    # user predict
    train_ratings_dict, compressed_test_ratings_dict, user_movie_ratings = get_user_rating_dicts(ratings, films)
    _, movies_all_genres_matrix = get_movies_aspect_matrix(films, "genre")
    _, movies_all_directors_matrix = get_movies_aspect_matrix(films, "director")
    _, movies_all_actors_matrix = get_movies_aspect_matrix(films, "actors")
    # compute user similarity
    ratings_dict, sims = compute_similarity(train_ratings_dict, films)
    # user prediction
    predictions = predictions(MUR, MUG, MUA, MUD, films,
                              compressed_test_ratings_dict, ratings_dict,
                              sims, movies_all_genres_matrix,
                              movies_all_directors_matrix,
                              movies_all_actors_matrix, movielens_data)
    # generate group
    groups = generate_group(predictions.keys())
    # aggregate group ratings
    group_prediction = aggregate_group_rating(predictions, groups)
