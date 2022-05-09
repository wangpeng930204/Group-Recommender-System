''' compute different measures to determine performance '''

from compute_strength import film_strength
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


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
                genre_predict_all[str(film_id)] = genreRating
                actor_predict_all[str(film_id)] = actorRating
                director_predict_all[str(film_id)] = directorRating
                individual_pre[str(film_id)] = filmRating
        all_pres[user_id] = (individual_pre, genre_predict_all, actor_predict_all, director_predict_all)
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
