import pickle
import time

from measures import predictions
from processing import preprocessing, remove_missing_film

MUR = 0.1
MUG = 0.6
MUA = 0.1
MUD = 0.1

if __name__ == "__main__":
    movielens_data = "100k"
    ratings = pickle.load(open("data/100k_benchmark_ratings.pkl", "rb"))
    films = pickle.load(open("data/100k_benchmark_films_movielens.pkl", "rb"))

    # remove from ratings the missing films (that were missing info and hence were discarded)
    ratings, films = remove_missing_film(films, ratings)




    films, ratings_dict, compressed_test_ratings_dict, sims, movies_all_genres_matrix, movies_all_directors_matrix, \
    movies_all_actors_matrix = preprocessing(ratings, films, movielens_data)

    start = time.time()

    nr_predictions, accuracy, rmse, mae, precision, recall, f1 = predictions(MUR, MUG, MUA, MUD, films,
                                                                             compressed_test_ratings_dict, ratings_dict,
                                                                             sims, movies_all_genres_matrix,
                                                                             movies_all_directors_matrix,
                                                                             movies_all_actors_matrix, movielens_data)

    # print results
    print("Number of user-items pairs: %d" % nr_predictions)
    print("Accuracy: %.2f " % accuracy)
    print("RMSE: %.2f" % rmse)
    print("MAE: %.2f" % mae)
    print("Precision: %.2f" % precision)
    print("Recall: %.2f" % recall)
    print("F1: %.2f" % f1)
    end = time.time()
    print("\nComputing strengths took %d seconds" % (end - start))
