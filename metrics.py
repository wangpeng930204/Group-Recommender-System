import numpy as np
from sklearn.metrics import ndcg_score


def _calculate_ndcg(user_true_ratings, user_est_ratings):
    """Calculate the NDCG at k metric for the user based on his/her obversed rating and his/her predicted rating.
    Args:
        user_true_ratings (ndarray): An array contains the predicted rating on the test set.
        user_est_ratings (ndarray): An array contains the obversed rating on the test set.
        k (int): the k metric.
    Returns:
        ndcg: the ndcg score for the user.
    """
    # Sort user ratings by estimated value
    user_true_ratings_order = user_true_ratings.argsort()[::-1]
    user_est_ratings_order = user_est_ratings.argsort()[::-1]

    ndcg = dcg(user_true_ratings, user_est_ratings_order) / dcg(user_true_ratings, user_true_ratings_order)

    return ndcg


def dcg(ratings, order):
    """ Calculate discounted cumulative gain.
    Args:
        ratings (ndarray): the rating of the user on the test set.
        order (ndarray): list of item id, sorted by the rating.
    Returns:
        float: the discounted cumulative gain of the user.
    """
    dcg = 0
    for ith, item in enumerate(order):
        dcg += ratings[item] / np.log2(ith + 2)

    return dcg

def ndcg_individual(compressed_test_ratings_dict, user_pred):
  nDCG = []
  for user_id, true_ratings in compressed_test_ratings_dict.items():
    true_r = []
    mid = []
    if true_ratings:
        for (film_id, str_rating) in true_ratings:
          true_r.append(int(str_rating))
          mid.append(film_id)

        pred_rating = [user_pred[str(user_id)][0][str(i)] for i in mid]
        ndcg = _calculate_ndcg(np.array(true_r), np.array(pred_rating))
        nDCG.append(ndcg)
  mean_nDCG = sum(nDCG)/len(nDCG)
  return mean_nDCG

def ndcg_group(compressed_test_ratings_dict, groups,
               group_pred_rating, group_recommendation, strategy):
    nDCG = []
    for user_id, true_ratings in compressed_test_ratings_dict.items():
        true_r = []
        mid = []
        if true_ratings:
            for (film_id, str_rating) in true_ratings:
                true_r.append(int(str_rating))
                mid.append(film_id)   #observe: index of mid and corresponding true_r are the same
            group = [k for k, v in groups.items() if user_id in v]
            pred_mid = group_recommendation[group[0]]
            if (strategy == "threshold"):
                pred_rating_filter = [group_pred_rating[user_id][pred_mid.index(i)] for i in mid if i in pred_mid]
                true_r = [true_r[mid.index(i)] for i in mid if i in pred_mid]  # filter the true rating following by observe
            else:
                pred_rating_filter = [group_pred_rating[group[0]][pred_mid.index(i)] for i in mid if i in pred_mid]
            ndcg = _calculate_ndcg(np.array(true_r), np.array(pred_rating_filter))
            nDCG.append(ndcg)
    mean_nDCG = sum(nDCG) / len(nDCG)
    return mean_nDCG
