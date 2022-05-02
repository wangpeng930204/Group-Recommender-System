import pickle
import time

from group import generate_group, group_recommendation
from measures import predictions
from metrics import ndcg_group
from processing import remove_missing_film, get_user_rating_dicts, get_movies_aspect_matrix, \
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
    groups = generate_group(predictions.keys(), group_scale = 10)

    group_pred_rating_thr, group_rec_threshold, group_exp_threshold = group_recommendation(predictions, groups, "threshold", 2.5)

    group_pred_rating_frq, group_rec_frequency, group_exp_frequency = group_recommendation(predictions, groups, "frequency", 2.5)

    group_pred_rating_avg, group_rec_average, group_exp_average = group_recommendation(predictions, groups, "average", 2)

    print("nDCG for threshold strategy:", ndcg_group(compressed_test_ratings_dict, groups, group_pred_rating_thr, group_rec_threshold, strategy="threshold"))
    print("nDCG for frequency strategy:", ndcg_group(compressed_test_ratings_dict, groups, group_pred_rating_frq, group_rec_frequency, strategy="frequency"))
    print("nDCG for average strategy:", ndcg_group(compressed_test_ratings_dict, groups, group_pred_rating_avg, group_rec_average, strategy="average"))
