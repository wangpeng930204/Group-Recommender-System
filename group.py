import numpy as np


def generate_group(user_ids, sim_users=None, group_size=6, random=True):
    if random:
        groups = randomly_form_group(user_ids, group_size)
    else:
        groups = form_group_similarly(sim_users, group_size)
    return groups


def form_group_similarly(sim_users, group_size):
    groups = dict()
    sim_users = sim_users.copy()
    group_numbers = int(len(sim_users.keys()) / group_size)
    id = 0
    used_user_count = set()
    for uid in sim_users.keys():
        gid = "g" + str(id)
        one_group = []
        count = 0
        full_size = False
        for i in range(len(sim_users[uid])):
            sim_uid, _ = sim_users[uid][i]  # start from the second similar user
            if sim_uid not in used_user_count:
                if count < group_size:
                    one_group.append(sim_uid)
                    used_user_count.add(sim_uid)
                else:
                    full_size = True
                count += 1
        if full_size:
            groups[gid] = one_group
            id += 1

    return groups


def randomly_form_group(user_ids, group_size):
    group_num = 0
    user_ids = list(user_ids).copy()
    group_numbers = int(len(user_ids) / group_size)
    groups = dict()
    for group_num in range(group_numbers):
        one_group = []
        for mn in range(group_size):
            index = np.random.randint(1, len(user_ids))
            one_group.append(user_ids[index])
            user_ids.pop(index)
        group_id = "g" + str(group_num)
        groups[group_id] = one_group
    groups["g" + str(group_num + 1)] = user_ids
    return groups


def aggregate_group_rating(films, user_predictions, groups, MUG, MUA, MUD):
    group_predictions, group_members_predictions = dict(), dict()
    for gid in groups.keys():
        # count how many  members in this group rated this aspect
        count_actor, count_genre, count_director = dict(), dict(), dict()
        film_rating, actor_rating, genre_rating, director_rating, member_rating = dict(), dict(), dict(), dict(), dict()
        for uid in groups[gid]:
            member_rating[uid] = user_predictions[uid]
            (fr, gr, ar, dr) = user_predictions[uid]
            actor_rating, count_actor = sum_rating(actor_rating, ar, count_actor)
            genre_rating, count_genre = sum_rating(genre_rating, gr, count_genre)
            director_rating, count_director = sum_rating(director_rating, dr, count_director)

        ave_actor_rating = average_all(actor_rating, count_actor)
        ave_genre_rating = average_all(genre_rating, count_genre)
        ave_director_rating = average_all(director_rating, count_director)

        group_predictions[gid] = group_film_strength(films, ave_genre_rating, ave_actor_rating,
                                                     ave_director_rating, MUG, MUA, MUD)
        group_members_predictions[gid] = member_rating
    return group_predictions, group_members_predictions

def aggregate_average(films, user_predictions, groups, MUG, MUA, MUD, baseline="False"):
    group_predictions, group_members_predictions = dict(), dict()
    for key in groups.keys():
        # count how many  members in this group rated this aspect
        count_actor, count_genre, count_director, count_film = dict(), dict(), dict(), dict()
        film_rating, actor_rating, genre_rating, director_rating, member_rating = dict(), dict(), dict(), dict(), dict()
        if baseline:
          for user in groups[key]:
            member_rating[user] = user_predictions[user]
            (fr, gr, ar, dr) = user_predictions[user]
            # film_rating, count_film = sum_rating(film_rating, fr, count_film)
            for film in fr.keys():
                if film in film_rating.keys():
                    count_film[film] += 1
                    film_rating[film] = fr[film] + film_rating[film]
                else:
                    count_film[film] = 1
                    film_rating[str(film)] = fr[film]          
          group_predictions[key] = average_all(film_rating, count_film)             
        else:
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
        # group_members_predictions[key] = member_rating
    return group_predictions

def least_Misery_aggregate(films, user_predictions, groups, MUG, MUA, MUD):
    group_predictions, group_members_predictions, baseline_film = dict(), dict(), dict()
    for gid in groups.keys():
        baseline_film[gid] = dict()
        least_actor_rating, least_genre_rating, least_director_rating, member_rating = dict(), dict(), dict(), dict()
        for uid in groups[gid]:
            member_rating[uid] = user_predictions[uid]
            (fr, gr, ar, dr) = user_predictions[uid]
            baseline_film[gid] = update_film_least_rating(baseline_film[gid], fr)
            least_actor_rating = update_aspects_least_rating(least_actor_rating, ar)
            least_genre_rating = update_aspects_least_rating(least_genre_rating, gr)
            least_director_rating = update_aspects_least_rating(least_actor_rating, dr)

        group_predictions[gid] = group_film_strength(films, least_genre_rating, least_actor_rating,
                                                     least_director_rating, MUG, MUA, MUD)
        group_members_predictions[gid] = member_rating
    return group_predictions, group_members_predictions, baseline_film


def update_aspects_least_rating(least_rating, to_update):
    for film in to_update.keys():
        for aspect in to_update[film].keys():
            if aspect in least_rating.keys():
                least_rating[aspect] = min(least_rating[aspect], to_update[film][aspect])
            else:
                least_rating[str(aspect)] = to_update[film][aspect]
    return least_rating


def update_film_least_rating(least_rating, to_update):
    for film in to_update.keys():
        if film in least_rating.keys():
            least_rating[film] = min(least_rating[film], to_update[film])
        else:
            least_rating[str(film)] = to_update[film]
    return least_rating


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
        for genre in films[film]['genre']:
            if genre in avgGenreRating.keys():
                sum_genre_rating += avgGenreRating[genre]
                count_genre += 1
        if count_genre > 0:
            average_genre_rating = sum_genre_rating / count_genre
        else:
            average_genre_rating = 0
        for actor in films[film]['actors']:
            if actor in avgActorRating.keys():
                sum_actor_rating += avgActorRating[actor]
                count_actor += 1
        if count_actor > 0:
            average_actor_rating = sum_actor_rating / count_actor
        else:
            average_actor_rating = 0
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


# old code
def group_recommendation(group_predictions, member_predictions, groups, strategy, threshold, films, MUG, MUA, MUD):
    g_rating, g_recommendation, g_explanation = dict(), dict(), dict()
    for g_id in group_predictions.keys():
        num_member = len(groups[g_id])
        film_rating = group_predictions[g_id]
        # average by num of group members
        film_rating = {k: v / num_member for k, v in film_rating.items()}
        # sorted by rating
        film_rating = dict(sorted(film_rating.items(), key=lambda item: item[1], reverse=True))

        sorted_film = list(film_rating.keys())
        sorted_rating = list(film_rating.values())
        g_recommendation[g_id] = sorted_film
        g_rating[g_id] = sorted_rating

    return g_rating, g_recommendation, g_explanation


def give_group_recommendation(group_predictions):
    g_rating, g_recommendation = dict(), dict()
    for g_id in group_predictions.keys():
        film_rating = group_predictions[g_id]
        # sorted by rating
        film_rating = dict(sorted(film_rating.items(), key=lambda item: item[1], reverse=True))
        sorted_film = list(film_rating.keys())
        sorted_rating = list(film_rating.values())
        g_recommendation[g_id] = sorted_film
        g_rating[g_id] = sorted_rating
    return g_rating, g_recommendation


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
