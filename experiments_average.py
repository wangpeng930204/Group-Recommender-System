import pickle
import time

import numpy as np

from group import generate_group, give_group_recommendation, aggregate_average
from measures import predictions
from metrics import ndcg_group, ndcg_individual
from processing import remove_missing_film, get_user_rating_dicts, get_movies_aspect_matrix, \
    compute_similarity
from matplotlib import pyplot as plt

MUR = 0.1
MUG = 0.6
MUA = 0.1
MUD = 0.1


def ndcg_experiments(userPredictions, films, group_size, sim_users, baseline="False", random_group="False"):
    groups = generate_group(userPredictions.keys(), sim_users, group_size=group_size, random=random_group)

    group_predictions = aggregate_average(films, userPredictions, groups, MUG, MUA, MUD, baseline)
    
    group_evaluation = 0
    g_rating_a, g_recommendation_a = give_group_recommendation(group_predictions)
    group_evaluation = ndcg_group(compressed_test_ratings_dict, groups, g_rating_a, g_recommendation_a)
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
    sim_avg = []
    sim_bl = []
    random_avg = []
    random_bl = []
    ax_value = []
    user_evaluation = ndcg_individual(compressed_test_ratings_dict, user_predictions) ##give individual ndcg here
    user_evaluations = []
    for group_scale in range(2, 10):
        user_evaluations.append(user_evaluation)
        ax_value.append(group_scale)
        ave_eva = ndcg_experiments(user_predictions, films, group_scale, sims, baseline="True", random_group="True")
        random_bl.append(ave_eva)
        ave_eva = ndcg_experiments(user_predictions, films, group_scale, sims, baseline="False", random_group="True")
        random_avg.append(ave_eva)
        ave_eva = ndcg_experiments(user_predictions, films, group_scale, sims, baseline="True", random_group="False")
        sim_bl.append(ave_eva)
        ave_eva = ndcg_experiments(user_predictions, films, group_scale, sims, baseline="False", random_group="False")
        sim_avg.append(ave_eva)

    barWidth = 0.2
    br1 = np.arange(len(user_evaluations))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    # Make the plot
    plt.bar(br1, random_bl, color='r', width=barWidth,
            edgecolor='grey', label='random baseline')
    plt.bar(br2, random_avg, color='g', width=barWidth,
            edgecolor='grey', label='random average strategy')
    plt.bar(br3, sim_avg, color='y', width=barWidth,
            edgecolor='grey', label='similar average strategy')
    plt.bar(br4, sim_bl, color='b', width=barWidth,
            edgecolor='grey', label='similar baseline')

    plt.title("The NDCG of Top 3 predicted film")
    plt.xlabel('Group Size', fontweight='bold', fontsize=15)
    plt.ylabel('NDCG Evaluation', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(user_evaluations))],
               ['2', '3', '4', '5', '6', '7', '8', '9'])
    plt.ylim([0.5, 1])
    plt.legend()
    plt.tight_layout()
    plt.show()

