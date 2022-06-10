import pickle

import numpy as np

from group import generate_group, least_Misery_aggregate, give_group_recommendation
from measures import predictions
from metrics import ndcg_group, ndcg_individual
from processing import remove_missing_film, get_user_rating_dicts, get_movies_aspect_matrix, \
    compute_similarity
from matplotlib import pyplot as plt

MUR = 0.1
MUG = 0.6
MUA = 0.1
MUD = 0.1


def ndcg_experiments(sorted_test, userPredictions, films, group_size):
    groups = generate_group(userPredictions.keys(), group_size=group_size)
    least_predictions, least_members_predictions, baseline = least_Misery_aggregate(films, userPredictions, groups, MUG,
                                                                                    MUA, MUD)
    g_rating_a, g_recommendation_a = give_group_recommendation(least_predictions)
    baseline_rating, baseline_ranking = give_group_recommendation(baseline)
    group_evaluation = ndcg_group(sorted_test, groups, g_rating_a, g_recommendation_a, "average")
    baseline_evaluation = ndcg_group(sorted_test, groups, baseline_rating, baseline_ranking, "average")
    return group_evaluation, baseline_evaluation


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
    evaluations_least_misery = []
    evaluations_baseline = []
    ax_value = []
    user_evaluation = ndcg_individual(compressed_test_ratings_dict, user_predictions)  # give individual ndcg here
    user_evaluations = []

    for group_scale in range(2, 10):
        user_evaluations.append(user_evaluation)
        ax_value.append(group_scale)
        lm_ndcg, baseline_ndcg = ndcg_experiments(compressed_test_ratings_dict, user_predictions, films, group_scale)
        evaluations_least_misery.append(lm_ndcg)
        evaluations_baseline.append(baseline_ndcg)

    barWidth = 0.25
    br1 = np.arange(len(user_evaluations))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    # Make the plot
    plt.bar(br1, user_evaluations, color='r', width=barWidth,
            edgecolor='grey', label='Individual')
    plt.bar(br2, evaluations_least_misery, color='y', width=barWidth,
            edgecolor='grey', label='Least Misery')
    plt.bar(br3, evaluations_baseline, color='b', width=barWidth,
            edgecolor='grey', label='Baseline')

    plt.title("The NDCG of Top 3 predicted film")
    plt.xlabel('Group Size', fontweight='bold', fontsize=15)
    plt.ylabel('NDCG Evaluation', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(user_evaluations))],
               ['2', '3', '4', '5', '6', '7', '8', '9'])
    plt.ylim([0.5, 1])
    plt.legend()
    plt.tight_layout()
    plt.show()
