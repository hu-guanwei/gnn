import numpy as np
from typing import List, Dict, Set


def sample_neighbors(src_nodes: List[int], num_neighbors: int, adj_map: Dict[int, Set[int]]) -> List[int]:
    ans = []

    for this_node in src_nodes:
        this_neighbors = list(adj_map[this_node])
        neighbor_sample = np.random.choice(a=this_neighbors, size=num_neighbors,
                                           replace=num_neighbors > len(this_neighbors))
        ans.extend(neighbor_sample)

    return ans


def multi_hop_sampling(src_nodes: List[int],
                       num_neighbor_list: List[int],
                       adj_map: Dict[int, Set[int]]) -> List[List[int]]:
    q = [src_nodes]

    for num_neighbors in num_neighbor_list:
        q.append(sample_neighbors(q[-1], num_neighbors, adj_map))

    return q


if __name__ == "__main__":
    adj_map = {0: {1, 2},
               1: {3},
               2: {0, 1},
               3: [0]}

    ans = sample_neighbors(src_nodes=[0, 1], num_neighbors=3, adj_map=adj_map)
    print(ans, type(ans))
    ans = multi_hop_sampling(src_nodes=[0, 1], num_neighbor_list=[3, 2], adj_map=adj_map)
    print(ans)
