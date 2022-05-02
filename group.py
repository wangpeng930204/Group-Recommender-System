# Todo find a way yo generate group for GRs
import operator

import numpy as np


def generate_group(user_ids, group_scale = 10, random=True):
    groups = []
    if random:
        groups = randomly_form_group(user_ids, group_scale)
        # train_ratings_dict['group'] = groups
    return groups


def randomly_form_group(user_ids, group_scale):
    user_ids_ = list(user_ids).copy()
    group_numbers = int(len(user_ids) / group_scale)
    groups = dict()
    for gn in range(group_numbers):
        one_group = []
        for mn in range(group_scale):
            index = np.random.randint(1, len(user_ids_))
            one_group.append(user_ids_[index])
            user_ids_.pop(index)
        group_id = "g" + str(gn)
        groups[group_id] = one_group
    groups["g"+str(gn+1)] = user_ids_
    return groups


def aggregate_group_rating(user_predictions, groups):
    group_predictions, group_members_predictions = dict(), dict()
    for key in groups.keys():
        count = 0
        film_rating, actor_rating, genre_rating, director_rating, member_rating = dict(), dict(), dict(), dict(), dict()
        for user in groups[key]:
            count += 1
            member_rating[user] = user_predictions[user]
            (fr, ar, gr, dr) = user_predictions[user]
            film_rating = sum_rating(film_rating, fr)
            actor_rating = sum_rating(actor_rating, ar)
            genre_rating = sum_rating(genre_rating, gr)
            director_rating = sum_rating(director_rating, dr)
        group_predictions[key] = (film_rating, actor_rating, genre_rating, director_rating)
        group_members_predictions[key] = member_rating
    return group_predictions, group_members_predictions


def sum_rating(base, to_sum):
    for key in to_sum:
        if key in base.keys():
            base[key] += to_sum[key]
        else:
            base[key] = to_sum[key]
    return base


def group_recommendation(user_predictions, groups, strategy, threshold):
    if strategy == "frequency":
        g_rating, g_recommendation, g_explanation = frequency_strategy(user_predictions, groups, threshold)
    else:
        g_rating, g_recommendation, g_explanation = dict(), dict(), dict()
        group_predictions, group_members_predictions = aggregate_group_rating(user_predictions, groups)
        for g_id in group_predictions.keys():
            num_member = len(groups[g_id])
            (film_rating, actor_rating, genre_rating, director_rating) = group_predictions[g_id]
            # average by num of group members
            film_rating = {k: v / num_member for k, v in film_rating.items()}
            # sorted by rating
            film_rating = dict(sorted(film_rating.items(), key=lambda item: item[1]))
            # average by num of group members
            actor_rating = {k: v / num_member for k, v in actor_rating.items()}
            # sorted by rating
            actor_rating = dict(sorted(actor_rating.items(), key=lambda item: item[1]))
            # average by num of group members
            genre_rating = {k: v / num_member for k, v in genre_rating.items()}
            # sorted by rating
            genre_rating = dict(sorted(genre_rating.items(), key=lambda item: item[1]))
            # average by num of group members
            director_rating = {k: v / num_member for k, v in director_rating.items()}
            # sorted by rating
            director_rating = dict(sorted(director_rating.items(), key=lambda item: item[1]))
            if strategy == "average":
                sorted_film = list(film_rating.keys())
                sorted_rating = list(film_rating.values())
                # if len(sorted_film) > 3:
                #     g_recommendation[g_id] = sorted_film[:3]
                # else:
                g_recommendation[g_id] = sorted_film
                g_rating[g_id] = sorted_rating
            elif strategy == "threshold":
                member_rating_filter, recommendation, dislike = strategy_threshold(film_rating, group_members_predictions[g_id], threshold)
                g_recommendation[g_id] = recommendation
                g_explanation[g_id] = dislike
                g_rating.update(member_rating_filter)
    return g_rating, g_recommendation, g_explanation


def strategy_threshold(group_pre, group_member_pre, threshold):
    recommendation = []
    members_like = dict()
    member_rating_filter = dict()
    for u_id in group_member_pre.keys():
        like = []
        rating_filter_list = []
        (film_rating, actor_rating, genre_rating, director_rating) = group_member_pre[u_id]
        for film in film_rating.keys():
            if film_rating[film] >= threshold:
                like.append(film)
                # group_like.update(film)
        
        members_like[u_id] = like
    # for film in group_pre.keys():
    #     if film in group_like:
    #         recommendation.append(film)
    for v in members_like.values():
      recommendation.append(v)
    recommendation_ = list(set.intersection(*[set(x) for x in recommendation]))
    for u_id in group_member_pre.keys():
      rating_filter_list = []  
      (film_rating, actor_rating, genre_rating, director_rating) = group_member_pre[u_id]
      for film in recommendation_:
        rating_filter_list.append(film_rating[film])
      member_rating_filter[u_id] = rating_filter_list
    return member_rating_filter, recommendation_, members_like


def frequency_strategy(user_predictions, groups, threshold):
    recommendation = dict()
    group_explanation = dict()
    group_frequency = dict()
    for g_id in groups.keys():
        loved_film_frequency = dict()
        user_explanation = dict()
        for u_id in groups[g_id]:
            (fr, ar, gr, dr) = user_predictions[u_id]
            dislike_film, like_film = rating_filter(fr, threshold)
            dislike_actor, like_actor = rating_filter(ar, threshold)
            dislike_genre, like_genre = rating_filter(gr, threshold)
            dislike_director, like_director = rating_filter(dr, threshold)
            user_explanation[u_id] = (dislike_film, like_film)
            for film in like_film:
                if film in loved_film_frequency.keys():
                    loved_film_frequency[film] += 1
                else:
                    loved_film_frequency[film] = 1
        loved_film_frequency = dict(sorted(loved_film_frequency.items(), key=lambda item: item[1]))
        recommendation[g_id] = list(loved_film_frequency.keys())
        group_frequency[g_id] = list(loved_film_frequency.values())
        group_explanation[g_id] = user_explanation

    return group_frequency, recommendation, group_explanation


def rating_filter(ratings, threshold):
    rv = list(ratings.values())
    dislike = set()
    like = set()
    for k in ratings.keys():
        if ratings[k] < threshold:
            dislike.update(k)
        else:
            like.update(k)
    return dislike, like
