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


def top5_calculate_ndcg(user_true_ratings, user_est_ratings):
    """Calculate the NDCG at k metric for the user based on his/her obversed rating and his/her predicted rating.
    Args:
        user_true_ratings (ndarray): An array contains the predicted rating on the test set.
        user_est_ratings (ndarray): An array contains the obversed rating on the test set.
        k (int): the k metric.
    Returns:
        ndcg: the ndcg score for the user.
    """
    # Sort user ratings by estimated value
    user_true_ratings_order = user_true_ratings.argsort()[::-1][:3]
    user_est_ratings_order = user_est_ratings.argsort()[::-1][:3]

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
    """ Calculate discounted cumulative gain for group.
    Args:
        compressed_test_ratings_dict (dictionary): the rating of the user on the test set.
        user_pred (dictionary): rating prediction of user
    Returns:
        float: the mean discounted cumulative gain of the all users.
    """
    nDCG = []
    for user_id, true_ratings in compressed_test_ratings_dict.items():
        true_r = []
        mid = []
        if true_ratings:
            for (film_id, str_rating) in true_ratings:
                true_r.append(int(str_rating))
                mid.append(film_id)

            pred_rating = [user_pred[str(user_id)][0][str(i)] for i in mid]
            # ndcg = _calculate_ndcg(np.array(true_r), np.array(pred_rating))
            ndcg = top5_calculate_ndcg(np.array(true_r), np.array(pred_rating))
            nDCG.append(ndcg)
    mean_nDCG = sum(nDCG) / len(nDCG)
    return mean_nDCG


def ndcg_group(compressed_test_ratings_dict, groups,
               group_pred_rating, group_recommendation):
    """ Calculate discounted cumulative gain for group.
    Args:
        compressed_test_ratings_dict (dictionary): the rating of the user on the test set.
        groups (dictionary): group id and its list of members
        group_pred_rating (dictionary): rating prediction of group
        group_recommendation (dictionary): movie id in top recommendation
    Returns:
        float: the mean discounted cumulative gain of all groups.
    """
    nDCG = []
    for user_id, true_ratings in compressed_test_ratings_dict.items():
        true_r = []
        mid = []
        if true_ratings:
            for (film_id, str_rating) in true_ratings:
                true_r.append(int(str_rating))
                mid.append(film_id)  # observe: index of mid and corresponding true_r are the same
            group = [k for k, v in groups.items() if user_id in v]
            if len(group) > 0:
                pred_mid = group_recommendation[group[0]]
                pred_rating_filter = [group_pred_rating[group[0]][pred_mid.index(i)] for i in mid if i in pred_mid]
                if len(true_r) > 2 and len(pred_rating_filter) > 2:
                    ndcg = top5_calculate_ndcg(np.array(true_r), np.array(pred_rating_filter))
                    nDCG.append(ndcg)
    if len(nDCG) == 0:
        mean_nDCG = 0
    else:
        mean_nDCG = sum(nDCG) / len(nDCG)
    return mean_nDCG
