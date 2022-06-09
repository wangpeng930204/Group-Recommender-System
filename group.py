# Todo find a way yo generate group for GRs
import operator

import numpy as np


def generate_group(user_ids, group_scale=6, random=True):
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
    groups["g" + str(gn + 1)] = user_ids_
    return groups


def aggregate_group_rating(films, user_predictions, groups, MUG, MUA, MUD):
    group_predictions, group_members_predictions = dict(), dict()
    for key in groups.keys():
        # count how many  members in this group rated this aspect
        count_actor, count_genre, count_director = dict(), dict(), dict()
        film_rating, actor_rating, genre_rating, director_rating, member_rating = dict(), dict(), dict(), dict(), dict()
        for user in groups[key]:
            member_rating[user] = user_predictions[user]
            (fr, gr, ar, dr) = user_predictions[user]
            # film_rating, count_film = sum_rating(film_rating, fr, count_film)
            actor_rating, count_actor = sum_rating(actor_rating, ar, count_actor)
            genre_rating, count_genre = sum_rating(genre_rating, gr, count_genre)
            director_rating, count_director = sum_rating(director_rating, dr, count_director)

        ave_actor_rating = average_all(actor_rating, count_actor)
        ave_genre_rating = average_all(genre_rating, count_genre)
        ave_director_rating = average_all(director_rating, count_director)

        group_predictions[key] = group_film_strength(films, ave_genre_rating, ave_actor_rating,
                                                     ave_director_rating, MUG, MUA, MUD)
        group_members_predictions[key] = member_rating
    return group_predictions, group_members_predictions


def sum_rating(base, to_sum, count):
    for film in to_sum.keys():
        for key in to_sum[film].keys():
            if key in base.keys():
                count[key] += 1
                base[key] = to_sum[film][key] + base[key]
            else:
                count[key] = 1
                base[str(key)] = to_sum[film][key]
    return base, count


def average_all(predict, count):
    for key in predict.keys():
        predict[key] = predict[key] / count[key]
    return predict


def group_film_strength(films, avgGenreRating, avgActorRating, avgDirectorRating, MUG, MUA, MUD):
    group_pre = dict()
    for film in films:
        sum_genre_rating, sum_director_rating, sum_actor_rating = 0, 0, 0
        count_genre, count_director, count_actor = 0, 0, 0
        average_genre_rating, average_actor_rating, average_director_rating = 0, 0, 0
        # if type(films[film]['genre']) is str:
        #     films[film]['genre'] = [films[film]['genre']]
        for genre in films[film]['genre']:
            if genre in avgGenreRating.keys():
                sum_genre_rating += avgGenreRating[genre]
                count_genre += 1
        if count_genre > 0:
            average_genre_rating = sum_genre_rating / count_genre
        else:
            average_genre_rating = 0
        # if type(films[film]['actors']) is str:
        #     films[film]['actors'] = [films[film]['actors']]
        for actor in films[film]['actors']:
            if actor in avgActorRating.keys():
                sum_actor_rating += avgActorRating[actor]
                count_actor += 1
        if count_actor > 0:
            average_actor_rating = sum_actor_rating / count_actor
        else:
            average_actor_rating = 0
        # if type(films[film]['director']) is str:
        #     films[film]['director'] = [films[film]['director']]
        for director in films[film]['director']:
            if director in avgDirectorRating.keys():
                sum_director_rating += avgDirectorRating[director]
                count_director += 1
        if count_director > 0:
            average_director_rating = sum_director_rating / count_director
        else:
            average_director_rating = 0
        item_strength = ((MUG * average_genre_rating) + (MUA * average_actor_rating) + (
                MUD * average_director_rating)) / (
                                MUG + MUA + MUD)
        group_pre[film] = (((item_strength + 1) * 2) + 1)

    return group_pre


def group_recommendation(group_predictions, member_predictions, groups, strategy, threshold, films, MUG, MUA, MUD):
    g_rating, g_recommendation, g_explanation = dict(), dict(), dict()
    for g_id in group_predictions.keys():
        num_member = len(groups[g_id])
        film_rating = group_predictions[g_id]
        # average by num of group members
        film_rating = {k: v / num_member for k, v in film_rating.items()}
        # sorted by rating
        film_rating = dict(sorted(film_rating.items(), key=lambda item: item[1], reverse=True))
        if strategy == "average":
            sorted_film = list(film_rating.keys())
            sorted_rating = list(film_rating.values())
            g_recommendation[g_id] = sorted_film
            g_rating[g_id] = sorted_rating
        elif strategy == "threshold":
            member_rating_filter, recommendation, dislike = strategy_threshold(film_rating, member_predictions[g_id],
                                                                               threshold)
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
            else:
                print(film_rating[film], u_id)
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
        loved_film_frequency = dict(sorted(loved_film_frequency.items(), key=lambda item: item[1], reverse=True))
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
