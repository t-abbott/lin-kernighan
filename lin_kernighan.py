"""
An implementation of the Lin-Kernighan TSP algorithm.
"""
from __future__ import annotations
from typing import Callable

import operator

from utils import find_random_tour, tour_cost
from utils import Matrix, Tour, Edge


class TabuList:
    """An edge tabu list supporting ordered insertion and O(1) lookup.

    TODO refactor __getitem__ to use chain depth
    """

    def __init__(self, items: list[Edge] | None = None) -> None:
        if not items:
            items = []

        self._item_list = items
        self._item_set = set(items)
        self.n_items = len(self._item_list)

        self.i = 0

    def __len__(self) -> int:
        return self.n_items

    def __getitem__(self, index: int | slice) -> Edge | TabuList:
        if isinstance(index, slice):
            items = self._item_list[index.start : index.stop : index.step]

            return TabuList(items)

        return self._item_list[index]

    def __add__(self, other: Edge) -> TabuList:
        return TabuList(self.edges + [other])

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i < self.n_items:
            item = self._item_list[self.i]
            self.i += 1

            return item

        raise StopIteration

    def __contains__(self, other: Edge) -> bool:
        """
        Tests if the edge `other` is in this tabu list
        """
        rev_other = (other[1], other[0])

        return other in self._item_set or rev_other in self._item_set

    def __str__(self):
        return "TabuList(" + str(self.edges) + ")"

    @property
    def edges(self) -> list[Edge]:
        """
        An ordered list of the edges in this tabu list
        """
        return self._item_list

    @property
    def edge_set(self) -> set[Edge]:
        """
        A set of the edges in this tabu list
        """
        return self._item_set

    def add(self, item: Edge) -> None:
        """
        Adds `item` to this tabu list
        """
        self._item_list.append(item)
        self._item_set.add(item)

        self.n_items += 1

    def depth_of(self, edge: Edge) -> int | None:
        """
        Returns the index (i.e. chain depth) of `edge` in this tabu list
        """
        rev_edge = (edge[1], edge[0])

        for i in range(self.n_items):
            if self.edges[i] == edge or self.edges[i] == rev_edge:
                return i + 1

        return None


def build_candidate_lists(dist_matrix: Matrix, n_candidates: int = 5) -> Matrix:
    """
    Build matrix of candiate neighbour lists for each city in `dist_matrix`.
    Candidate cities are ranked by shortest distance.
    """

    snd = operator.itemgetter(1)  # snd x y = y
    candidates = [None] * len(dist_matrix)

    for city, neighbour_dists in enumerate(dist_matrix):
        # sort neighbours by their distance to `city`
        sorted_neighbours = sorted(enumerate(neighbour_dists), key=snd)

        # map `city` the first `n_candidates` closest neighbours
        candidates[city] = [
            neighbour for (neighbour, _) in sorted_neighbours[1 : n_candidates + 1]
        ]

    return candidates


def build_neighbour_list(tour: Tour) -> list[tuple[int, int]]:
    """Build a list mapping cities in `tour` to the cities
    either side of it.
    """

    n_cities = len(tour)
    neighbours = [None] * n_cities

    for i, city in enumerate(tour):
        left = (i - 1) % n_cities
        right = (i + 1) % n_cities

        neighbours[city] = (tour[left], tour[right])

    return neighbours


def edges_of_tour(tour: Tour) -> list[Edge]:
    """Converts a tour `tour` into a list of edges."""
    n_cities = len(tour)
    return [(tour[i], tour[(i + 1) % n_cities]) for i in range(len(tour))]


def tour_of_edges(edges: list[Edge]) -> Tour:
    """Converts a list of tour edges `edges` into a list of cities."""
    tour = [-1] * len(edges)
    for i, edge in enumerate(edges):
        tour[i] = edge[0]

    return tour


# -----


def reverse_edge(edge: Edge) -> Edge:
    """
    .
    """
    return (edge[1], edge[0])


def order_edges(edges: list[Edge]) -> Tour | None:
    """
    Returns the list of edges `edges` ordered into a valid tour.
    Returns `None` if `edges` contains a closed cycle induced
    by bad choices of `x_i`.

    Adding and deleting edges `x_i`, `y_i` implicitly reverses arcs
    along T, meaning we can't simply iterate over the edges looking
    for successors. By flipping all edges when we reach the end of
    one subtour we can construct the reversed subtour following after
    it.
    """

    n_edges = len(edges)
    start_edge = edges[0]
    edges = edges.copy()

    seen = TabuList([start_edge])
    prev_edge = start_edge
    next_city = start_edge[1]

    completed_tour = False
    while not completed_tour:
        found_successor = False

        for edge in edges:
            if edge == prev_edge or reverse_edge(edge) == prev_edge:
                continue

            if edge[0] == next_city:
                if len(seen) == n_edges and next_city == start_edge[0]:
                    completed_tour = True
                    break

                if edge in seen:
                    # `edges` contains a disconnected cycle
                    return None
                else:
                    found_successor = True

                next_city = edge[1]
                prev_edge = edge
                seen.add(edge)

        # flip edges to find the inverted subtour starting at
        # `next_city`
        if not found_successor:
            edges = list(map(reverse_edge, edges))

    return seen.edges


def build_tour(
    tour: Tour,
    removed_edges: TabuList,
    added_edges: TabuList,
) -> Tour | None:
    """
    Attempt to build a new tour based on `tour` and ...
    """
    old_edges = edges_of_tour(tour)
    new_edges = [
        edge for edge in old_edges if edge not in removed_edges
    ] + added_edges.edges

    edges = order_edges(new_edges)
    if edges is None:
        return None
    else:
        tour = tour_of_edges(edges)
        return tour


def kopt_move(
    base_tour: Tour,
    x_1: Edge,
    y_1: Edge,
    candidate_list: Matrix,
    neighbour_list: list[tuple[int, int]],
    cost: Callable[[Edge], int],
) -> tuple[Tour, int]:
    """
    Performs a single k-opt move on `base_tour` from initial tabu edges `x_1` and `y_1`
    """
    n_cities = len(base_tour)
    best_tour = base_tour
    current_gain = cost(x_1) - cost(y_1)
    best_gain = 0

    i = 1
    k = i
    y_k_final = (x_1[1], x_1[0])

    first_city = x_1[0]

    x_prev = x_1
    y_prev = y_1

    # initialise tabu lists
    X = TabuList([x_1])
    Y = TabuList([y_1])

    #
    # Step 4 - choose `x_i`, `y_i`
    #
    #   x_i = (t_k, t_l)
    #   y_i = (t_l, t_m)
    #
    while len(X) <= n_cities / 2:
        i += 1

        # Find the `x_i \in T` determined by `y_prev`
        city_k = y_prev[1]
        city_l = None
        x_i = None
        g_final = None
        new_tour = None

        #
        # Step 4(a) - choose `x_i` such that deleting `x_i` doesn't create a closed tour
        #
        for potential_city in neighbour_list[city_k]:
            potential_x_i = (city_k, potential_city)
            potential_y_final = (potential_city, first_city)
            potential_g_final = cost(potential_x_i) - cost(potential_y_final)

            potential_tour = build_tour(
                base_tour,
                X + potential_x_i,
                Y + potential_y_final,
            )

            if potential_tour is None:
                continue
            else:
                city_l = potential_city
                x_i = potential_x_i
                g_final = potential_g_final
                new_tour = potential_tour
                break

        if x_i is None:
            # No valid choice of `x_i`; backtrack
            break

        #
        # Step 4(c) - ensure X, Y are disjoint and `x_i` not previously deleted
        #
        if x_i in X or x_i in Y:
            break

        X.add(x_i)

        #
        # Step 4(f) - check to see if we can form an improving tour from `x_i`
        #
        if current_gain + g_final > best_gain:
            k = i
            best_gain = current_gain + g_final
            best_tour = new_tour

        # Choose `y_i`
        y_i = None
        gain_i = None

        for potential_city in candidate_list[city_l]:

            # check `y_i` not in `tour`
            if potential_city in neighbour_list[city_l]:
                continue
            elif potential_city == first_city:
                continue

            potential_y_i = (city_l, potential_city)
            potential_gain = cost(x_i) - cost(potential_y_i)

            #
            # Step 4(c) - ensure X, Y are disjoint
            #
            if potential_y_i in X or potential_y_i in Y:
                continue

            #
            # Step 4(d) - gain criterion
            #
            if current_gain + potential_gain <= 0:
                continue

            y_i = potential_y_i
            gain_i = potential_gain

        if y_i is None:
            # No valid choice of `y_i`; backtrack
            break

        current_gain += gain_i
        Y.add(y_i)

        x_prev = x_i
        y_prev = y_i

    return best_tour, best_gain


def lin_kernighan(
    initial_tour: Tour,
    dist_matrix: Matrix,
) -> Tour:
    """
    .
    """

    def cost(edge: Edge, dist_matrix: Matrix = dist_matrix) -> int:
        """
        Helper function to return the cost of `edge` in `dist_matrix`
        """
        return dist_matrix[edge[0]][edge[1]]

    # Step 1 - start with an initial tour
    best_tour = initial_tour
    best_weight = tour_cost(initial_tour, dist_matrix)
    n_cities = len(initial_tour)
    candidate_list = build_candidate_lists(dist_matrix)
    seen_tours: set[Tour] = set()

    improved_tour = True
    duplicate_tour = False

    neighbour_list = build_neighbour_list(best_tour)

    while improved_tour:
        improved_tour = False

        # Step (2) - choose a first edge `x_1` to delete
        for city_1 in range(n_cities):

            for city_2 in neighbour_list[city_1]:
                if improved_tour or duplicate_tour:
                    break

                # Step (3) - choose a first edge `y_1` to add
                for city_3 in candidate_list[city_2]:
                    if improved_tour or duplicate_tour:
                        break

                    # ignore `y_1` edges currently in `tour`
                    if city_3 in neighbour_list[city_2]:
                        continue

                    # apply gain criterion for `x_1`, `y_1`
                    x_1 = (city_1, city_2)
                    y_1 = (city_2, city_3)

                    made_kopt_move = True

                    while made_kopt_move:
                        made_kopt_move = False

                        #
                        # Step 4 - attempt k-opt moves
                        #
                        new_tour, gain = kopt_move(
                            best_tour,
                            x_1,
                            y_1,
                            candidate_list,
                            neighbour_list,
                            cost,
                        )

                        if gain == 0:
                            break

                        tour_tuple = tuple(new_tour)
                        if tour_tuple in seen_tours:
                            duplicate_tour = True
                            break

                        # memoise the resulting tour
                        seen_tours.add(tour_tuple)

                        # record improving moves
                        if gain > 0:
                            best_tour = new_tour
                            best_weight -= gain

                            improved_tour = True
                            made_kopt_move = True
                            neighbour_list = build_neighbour_list(best_tour)

                        # why?
                        break

    return best_tour, best_weight


def find_tour(dist_matrix: Matrix) -> tuple[Tour, int]:
    """
    Returns a tour on `dist_matrix` and it's length
    """
    initial_tour = find_random_tour(dist_matrix)

    return lin_kernighan(initial_tour, dist_matrix)
