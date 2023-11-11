from collections import defaultdict
from itertools import combinations



def BFS_(graph: list[list[int]], s: int, t: int, parent: list[int]) -> bool:
    length = len(graph)
    print(graph)
    # Обход в ширину
    visited = [False] * (length)
    queue = []
    queue.append(s)
    visited[s] = True

    while queue:

        u = queue.pop(0)
        for ind, val in enumerate(graph[u]):
            if not visited[ind] and val > 0:
                queue.append(ind)
                visited[ind] = True
                parent[ind] = u
                if ind == t:
                    return True
    return False





def BFS(graph: list[list[int]], s: int, t: int, parent: list[int]) -> bool:
    length = len(graph)

    # Обход в ширину
    visited = [False] * (length)
    queue = []
    queue.append(s)
    visited[s] = True

    while queue:

        u = queue.pop(0)
        for ind, val in enumerate(graph[u]):
            if not visited[ind] and val > 0:
                queue.append(ind)
                visited[ind] = True
                parent[ind] = u
                if ind == t:
                    return True
    return False


def ford_fulkerson_algorithm(lab, vertex_source='S', vertex_dest='T', image_create: bool = True) -> set[
    int, list[list[str]]]:
    length = len(lab.graph.nodes)
    vertex_source = lab.renamer.correction_function(vertex_source)
    vertex_dest = lab.renamer.correction_function(vertex_dest)



    mat = lab.adj_matrix

    # This array is filled by BFS and to store path
    parent = [-1] * (length)

    max_flow = 0  # There is no flow initially
    i = 0
    # Этап 1 ищем путь из истока в сток по пути по которому можно пустить поток
    paths = []
    while BFS(mat, vertex_source, vertex_dest, parent):
        i += 1

        # в parent оказывается этот путь в формате parent[вершина] = откуда в эту вершину идет стрелка

        # begin
        # здесь находим максимально возможный поток по этому пути. поиск идет с конца.
        path_flow = float("Inf")
        s = vertex_dest
        while (s != vertex_source):
            path_flow = min(path_flow, mat[parent[s]][s])
            s = parent[s]
        # end

        max_flow += path_flow
        # begin
        # Здесь обновляем значения графа уменьшая значение "сколько всего можно пустить через соединение"
        # Уменьшение нужно для того чтобы поиск пути не проходил по лишним
        # И увеличивая текущий поток.
        v = vertex_dest
        while (v != vertex_source):
            u = parent[v]
            mat[u][v] -= path_flow
            mat[v][u] += path_flow
            lab._update_graph_in_ford_fulkerson_algorithm(u, v, path_flow)

            v = parent[v]
        # end
        path = lab._get_path_in_ford_fulkerson_algorithm(vertex_source, vertex_dest, parent)
        paths.append(path)
        if image_create:
            lab.save_graph_image('steps\\'+str(i) + "_1.png")
            lab.save_graph_image('steps\\'+str(i) + "_2.png", path)

    for edge in lab.graph.edges:
        lab.graph[edge[0]][edge[1]]["current_flow"] = 0
            #lab.graph[(lab.renamer(v1),lab.renamer(v2))]["current_flow"] = 0
    return max_flow, paths

def floyd_warshall_algo(G):

    dist = defaultdict(lambda: defaultdict(lambda: float("Inf")))

    for u in G:
        dist[u][u] = 0
    pred = defaultdict(dict)

    for u, v, d in G.edges(data=True):
        e_weight = d.get("max_flow", 1.0)
        dist[u][v] = min(e_weight, dist[u][v])
        pred[u][v] = u

    for w in G:
        dist_w = dist[w]
        for u in G:
            dist_u = dist[u]
            for v in G:
                d = dist_u[w] + dist_w[v]
                if dist_u[v] > d:
                    dist_u[v] = d
                    pred[u][v] = pred[w][v]

    res = []
    pairs = list(combinations(G.nodes, 2))
    for pair in pairs:
        prev = pred[pair[0]]
        curr = prev[pair[1]]
        path = [pair[1], curr]
        while curr != pair[0]:
            curr = prev[curr]
            path.append(curr)
        temp = list(reversed(path))

        res.append((temp, dist[pair[0]][pair[1]]))

    return res