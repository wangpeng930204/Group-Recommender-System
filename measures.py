''' compute different measures to determine performance '''

import jsonlines
from math import sqrt
from compute_strength import film_strength
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# Todo: find the group strategy for GRs
def group_strategy():
    print("Todo: find the group strategy for GRs")


# Todo: find the group prediction for GRs
def group_predictions(MUR, MUG, MUA, MUD, films, compressed_test_ratings_dict, ratings_dict, sims,
                      movies_all_genres_matrix,
                      movies_all_directors_matrix, movies_all_actors_matrix, data_origin, groups):
    group_pds = []
    for group in groups:
        for user in group:
            group_strategy()

    return group_pds


def predictions(MUR, MUG, MUA, MUD, films, compressed_test_ratings_dict, ratings_dict, sims, movies_all_genres_matrix,
                movies_all_directors_matrix, movies_all_actors_matrix, data_origin):
    # compute strengths
    all_pres = dict()
    for user_id, true_ratings in compressed_test_ratings_dict.items():
        individual_pre, genre_predict_all, actor_predict_all, director_predict_all = dict(), dict(), dict(), dict()

        if true_ratings:
            for (film_id, str_rating) in true_ratings:
                filmRating, genreRating, actorRating, directorRating = film_strength(MUR, MUG, MUA, MUD, user_id,
                                                                                     film_id, films, ratings_dict,
                                                                                     sims[user_id],
                                                                                     movies_all_genres_matrix,
                                                                                     movies_all_directors_matrix,
                                                                                     movies_all_actors_matrix)
                genre_predict_all.update(genreRating)
                actor_predict_all.update(actorRating)
                director_predict_all.update(directorRating)
                individual_pre[str(film_id)] = filmRating
                # if data_origin == 'netflix':
                #     predictions.append((int(str_rating), filmRating))
                # elif data_origin == 'small':
                #     predictions.append((float(str_rating), filmRating))
                # elif data_origin == '100k':
                #     predictions.append((int(str_rating), filmRating))
        all_pres[user_id] = (individual_pre, genre_predict_all, actor_predict_all, director_predict_all)
    # if data_origin == 'netflix':
    #     true_ratings = [x for (x, y) in predictions]
    #     predicted_ratings = [round(y) for (x, y) in predictions]
    #     p, r, f = binary_predictions(true_ratings, predicted_ratings)
    #     return len(predictions), arg_accuracy_int(predictions), sqrt(
    #         mean_squared_error(true_ratings, predicted_ratings)), mean_absolute_error(true_ratings,
    #                                                                                   predicted_ratings), p, r, f
    # elif data_origin == 'small':
    #     true_ratings = [x for (x, y) in predictions]
    #     predicted_ratings = [round_of_rating(y) for (x, y) in predictions]
    #     p, r, f = binary_predictions(true_ratings, predicted_ratings)
    #     return len(predictions), arg_accuracy_float(predictions), sqrt(
    #         mean_squared_error(true_ratings, predicted_ratings)), mean_absolute_error(true_ratings,
    #                                                                                   predicted_ratings), p, r, f
    # elif data_origin == '100k':
    #     true_ratings = [x for (x, y) in predictions]
    #     predicted_ratings = [round(y) for (x, y) in predictions]
    #     p, r, f = binary_predictions(true_ratings, predicted_ratings)
    #     return predictions, arg_accuracy_int(predictions), sqrt(
    #         mean_squared_error(true_ratings, predicted_ratings)), mean_absolute_error(true_ratings,
    #                                                                                   predicted_ratings), p, r, f
    return all_pres


def binary_predictions(true_ratings, predicted_ratings):
    assert len(true_ratings) == len(predicted_ratings)
    binary_true_ratings = []
    binary_predicted_ratings = []

    # make 3 a positive
    for i in range(len(true_ratings)):
        if true_ratings[i] >= 3:
            binary_true_ratings.append(1)
        else:
            binary_true_ratings.append(0)

        if predicted_ratings[i] >= 3:
            binary_predicted_ratings.append(1)
        else:
            binary_predicted_ratings.append(0)

    return precision_score(binary_true_ratings, binary_predicted_ratings), \
           recall_score(binary_true_ratings, binary_predicted_ratings), \
           f1_score(binary_true_ratings, binary_predicted_ratings)


def arg_accuracy_int(true_and_predicted_ratings):
    total_nr = len(true_and_predicted_ratings)
    total_pred = 0
    for i in range(total_nr):
        (true_rating, pred_rating) = true_and_predicted_ratings[i]
        if int(true_rating) - 1 <= round(pred_rating) <= int(true_rating) + 1:
            total_pred += 1

    return float(total_pred) / total_nr


def arg_accuracy_float(true_and_predicted_ratings):
    total_nr = len(true_and_predicted_ratings)
    total_pred = 0
    for i in range(total_nr):
        (true_rating, pred_rating) = true_and_predicted_ratings[i]
        if float(true_rating) - 1 <= round_of_rating(pred_rating) <= float(
                true_rating) + 1:
            total_pred += 1

    return float(total_pred) / total_nr


# round to nearest .5
def round_of_rating(number):
    return round(number * 2) / 2
