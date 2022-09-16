"""
TSP utility functions and types
"""

from random import shuffle

#
# Types
#


Matrix = list[list[int]]
Tour = list[int]
Edge = tuple[int, int]

#
# Utility functions
#


def is_connected(dist_matrix):
    """
    .
    """
    n_cities = len(dist_matrix)

    return all(map(lambda l: l == n_cities, map(len, dist_matrix)))


def edge_cost(c_1, c_2, dist_matrix):
    """
    .
    """
    return dist_matrix[c_1][c_2]


def reverse_sublist(lst, l, r):
    """
    Reverse the elements of `lst` between indices `l` and `r`
    """

    return lst[:l] + lst[l : r + 1][::-1] + lst[r + 1 :]


def reverse_subtour(tour, city_l, city_r):
    """
    Reverse the subtour of `tour` between cities `city_l` and `city_r`
    """
    i = tour.index(city_l) + 1
    j = tour.index(city_r)

    if i > j:
        i, j = j, i

    return tour[:i] + tour[i : j + 1][::-1] + tour[j + 1 :]


def tour_cost(tour, dist_matrix):
    """
    Find the cost of `tour` on `dist_matrix`
    """
    cost = 0

    for i in range(len(tour) - 1):
        cost += dist_matrix[tour[i]][tour[i + 1]]

    cost += dist_matrix[tour[-1]][tour[0]]

    return cost


def dist_dict_to_matrix(dist_dict):
    """
    Convert a dict to a matrix
    """
    return [dist_dict[i] for i in range(len(dist_dict))]


#
# Tour functions
#


def find_random_tour(dist_matrix):
    """
    Generate a random tour on the distance matrix `dist_matrix`.
    """
    n_cities = len(dist_matrix)

    tour = list(range(n_cities))
    shuffle(tour)

    return tour


def find_greedy_tour(dist_matrix, start_city: int = 0):
    """
    .
    """
    n_cities = len(dist_matrix)
    tour = [start_city]

    while len(tour) < n_cities:
        best = (float("inf"), None)

        # extend to the next closest city
        for city, distance in enumerate(dist_matrix[tour[-1]]):
            if city in tour:
                pass
            else:
                if distance < best[0]:
                    best = (distance, city)

        tour.append(best[1])

    return tour
