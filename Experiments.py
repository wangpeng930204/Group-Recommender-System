import pickle
import time

import numpy as np

from group import generate_group, group_recommendation, aggregate_group_rating
from measures import predictions
from metrics import ndcg_group, _calculate_ndcg
from processing import remove_missing_film, get_user_rating_dicts, get_movies_aspect_matrix, \
    compute_similarity
from matplotlib import pyplot as plt

MUR = 0.1
MUG = 0.6
MUA = 0.1
MUD = 0.1


def ndcg_experiments(userPredictions, films, group_scale, strategy):
    groups = generate_group(userPredictions.keys(), group_scale=group_scale)
    group_predictions, group_members_predictions = aggregate_group_rating(films, userPredictions, groups, MUG, MUA, MUD)
    group_evaluation = 0
    if strategy == "threshold":
        g_rating_t, g_recommendation_t, g_explanation_t = group_recommendation(group_predictions,
                                                                               group_members_predictions, groups,
                                                                               "threshold", 2.5,
                                                                               films, MUG, MUA, MUD)
        group_evaluation = ndcg_group(compressed_test_ratings_dict, groups, g_rating_t, g_recommendation_t, "threshold")
    elif strategy == "average":
        g_rating_a, g_recommendation_a, g_explanation_a = group_recommendation(group_predictions,
                                                                               group_members_predictions,
                                                                               groups, "average",
                                                                               2, films, MUG, MUA, MUD)
        group_evaluation = ndcg_group(compressed_test_ratings_dict, groups, g_rating_a, g_recommendation_a, "average")
    return group_evaluation


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
    user_predictions = predictions(MUR, MUG, MUA, MUD, films,
                                   compressed_test_ratings_dict, ratings_dict,
                                   sims, movies_all_genres_matrix,
                                   movies_all_directors_matrix,
                                   movies_all_actors_matrix, movielens_data)
    evaluations_average = []
    evaluation_threshold = []
    ax_value = []
    user_evaluation = _calculate_ndcg(compressed_test_ratings_dict, user_predictions) ##give individual ndcg here
    user_evaluations = []
    for group_scale in range(2, 10):
        user_evaluations.append(user_evaluation)
        ax_value.append(group_scale)
        ave_eva = ndcg_experiments(user_predictions, films, group_scale, "average")
        evaluations_average.append(ave_eva)
        ave_eva = ndcg_experiments(user_predictions, films, group_scale, "threshold")
        evaluation_threshold.append(ave_eva)
    plt.plot(evaluations_average)
    plt.title("NDCG Evaluation of group scales-strategy a")
    plt.show()
    plt.plot(evaluation_threshold)
    plt.title("NDCG Evaluation of group scales-strategy b")
    plt.show()

    data = [[30, 25, 50, 20],
            [40, 23, 51, 17],
            [35, 22, 45, 19]]
    X = np.arange(4)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.bar(X + 0.00, user_evaluation, color='b', width=0.25)
    ax.bar(X + 0.25, evaluations_average, color='g', width=0.25)
    ax.bar(X + 0.50, evaluation_threshold, color='r', width=0.25)
