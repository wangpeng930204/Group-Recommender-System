# Todo find a way yo generate group for GRs
import numpy as np


def generate_group(user_ids, random=True):
    groups = []
    if random:
        groups = randomly_form_group(user_ids)
        # train_ratings_dict['group'] = groups
    return groups


def randomly_form_group(user_ids):
    user_ids = list(user_ids)
    group_scale = 10
    group_numbers = int(len(user_ids) / 10)
    groups = dict()
    for gn in range(group_numbers):
        one_group = []
        for mn in range(np.random.randint(1, group_scale)):
            index = np.random.randint(1, len(user_ids))
            one_group.append(user_ids[index])
        group_id = "g" + str(gn)
        groups[group_id] = one_group
    return groups


def aggregate_group_rating(user_predictions, groups):
    group_predictions = dict()
    for key in groups.keys():
        count = 0
        film_rating, actor_rating, genre_rating, director_rating = dict(), dict(), dict(), dict()
        for user in groups[key]:
            count += 1
            (fr, ar, gr, dr) = user_predictions[user]
            film_rating = sum_rating(film_rating, fr)
            actor_rating = sum_rating(film_rating, ar)
            genre_rating = sum_rating(film_rating, gr)
            director_rating = sum_rating(film_rating, dr)
        group_predictions[key] = (film_rating, actor_rating, genre_rating, director_rating)
        group_predictions[key] = (film_rating, actor_rating, genre_rating, director_rating)
    return group_predictions


def sum_rating(base, to_sum):
    for key in to_sum:
        if key in base.keys():
            base[key] += to_sum[key]
        else:
            base[key] = to_sum[key]
    return base
