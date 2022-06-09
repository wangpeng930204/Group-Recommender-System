import pickle
import time

import numpy as np

from group import generate_group, group_recommendation, aggregate_group_rating
from measures import predictions
from metrics import ndcg_group, ndcg_individual
from processing import remove_missing_film, get_user_rating_dicts, get_movies_aspect_matrix, \
    compute_similarity
from matplotlib import pyplot as plt

MUR = 0.1
MUG = 0.6
MUA = 0.1
MUD = 0.1


def ndcg_experiments(sorted_test, userPredictions, films, group_scale, strategy):
    groups = generate_group(userPredictions.keys(), group_scale=group_scale)
    group_predictions, group_members_predictions = aggregate_group_rating(films, userPredictions, groups, MUG, MUA, MUD)
    group_evaluation = 0
    if strategy == "threshold":
        g_rating_t, g_recommendation_t, g_explanation_t = group_recommendation(group_predictions,
                                                                               group_members_predictions, groups,
                                                                               "threshold", 2,
                                                                               films, MUG, MUA, MUD)
        group_evaluation = ndcg_group(sorted_test, groups, g_rating_t, g_recommendation_t, "threshold")
    elif strategy == "average":
        g_rating_a, g_recommendation_a, g_explanation_a = group_recommendation(group_predictions,
                                                                               group_members_predictions,
                                                                               groups, "average",
                                                                               2.5, films, MUG, MUA, MUD)
        group_evaluation = ndcg_group(sorted_test, groups, g_rating_a, g_recommendation_a, "average")
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
    # sorted_test_ratings = dict()
    # for uid in compressed_test_ratings_dict.keys():
    #     sorted_test_ratings[uid] = sorted(compressed_test_ratings_dict[uid], key=lambda tup: tup[1], reverse=True)
    evaluations_average = []
    evaluation_threshold = []
    ax_value = []
    user_evaluation = ndcg_individual(compressed_test_ratings_dict, user_predictions)  ##give individual ndcg here
    user_evaluations = []
    for group_scale in range(2, 10):
        user_evaluations.append(user_evaluation)
        ax_value.append(group_scale)
        ave_eva = ndcg_experiments(compressed_test_ratings_dict, user_predictions, films, group_scale, "average")
        evaluations_average.append(ave_eva)
        ave_eva = ndcg_experiments(compressed_test_ratings_dict, user_predictions, films, group_scale, "threshold")
        evaluation_threshold.append(ave_eva)

    plt.plot(evaluations_average, label="Default setting")
    plt.plot(evaluation_threshold, label='Approval Voting')
    plt.plot(user_evaluations, label="Individual")
    plt.legend()
    plt.xlabel("Group Size")
    plt.title("NDCG Evaluation of group scales-strategy b")
    xvalues = np.arange(2, 10)
    plt.xticks(xvalues)
    plt.show()

    barWidth = 0.25
    br1 = np.arange(len(user_evaluations))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, user_evaluations, color='r', width=barWidth,
            edgecolor='grey', label='Individual')
    plt.bar(br2, evaluations_average, color='y', width=barWidth,
            edgecolor='grey', label='Default setting')
    plt.bar(br3, evaluation_threshold, color='b', width=barWidth,
            edgecolor='grey', label='Filtered')

    plt.title("The NDCG of Top 3 predicted film")
    # Adding Xticks
    plt.xlabel('Group Size', fontweight='bold', fontsize=15)
    plt.ylabel('NDCG Evaluation', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(user_evaluations))],
               ['2', '3', '4', '5', '6', '7', '8', '9'])
    plt.ylim([0.5, 1])
    # plt.legend(bbox_to_anchor=(0.95, 1.0), loc='upper left')
    plt.legend()
    plt.tight_layout()
    plt.show()
