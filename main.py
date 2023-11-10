import networkx as nx

import matplotlib.pyplot as plt

import numpy as np

from algorithms import ford_fulkerson_algorithm, floyd_warshall_algo
from renamer import Renamer
from itertools import combinations


def get_last_digit(line: str) -> int:
    for c in line[::-1]:
        if c.isnumeric():
            return int(c)
    raise AttributeError("in str must be a digit")


class First_lab:
    def create_report(self):
        print("Результат работы алгоритма ФФ: максимальный поток и простые цепи")
        print(ford_fulkerson_algorithm(self))
        print("Диаметр =", self.diam)
        print("ST-отделяющее множество =", self.separating_set)
        print("ST-разделяющее множество =", self.dividing_set)
        #self.draw_cut(self.dividing_set)


    def cut_creator(self):
        _, temp = ford_fulkerson_algorithm(self, image_create=False)
        chains = []
        for el in temp:
            t = []
            for i in range(len(el) - 1):
                t.append((el[i], el[i+1]))
            chains.append(t)
        cuts_dict = {}
        cuts_dict_ = {}
        for i in range(2, len(self.dividing_set)):
            cuts_dict[i] = list(combinations(self.dividing_set, i))
        for i in cuts_dict:
            for cut in cuts_dict[i]:
                cc = 0
                tt = True
                for edge_cut in cut:


                    for chain in chains:
                        if edge_cut in chain:
                            cc+=1
                            for edge_cut2 in cut:
                                if edge_cut2 != edge_cut and edge_cut2 in chain:
                                    tt = False
                if tt and cc == len(chains):
                    if i not in cuts_dict_.keys():
                        cuts_dict_[i] = []
                    cuts_dict_[i].append(cut)


        for i in cuts_dict_:
            for c in cuts_dict_[i]:
                print(c)
                self.draw_cut(c, str(c))
    def draw_cut(self, cut, fn = "test.png"):

        temp = [(u, v) for u, v in self.graph.edges()]
        for edge in cut:
            temp.pop(temp.index(edge))

        self.save_graph_image(fn, edges=temp, colored_edges=cut)


    def __init__(self):
        self._var = None
        self._number_group = None
        self._graph = None
        self._source = None
        self._destination = None
        self._renamer = Renamer()

    @property
    def renamer(self) -> Renamer:
        return self._renamer

    @property
    def var(self) -> int:
        return self._var

    @var.setter
    def var(self, var: int) -> None:
        if self._var is None:
            self._var = var
        else:
            raise AttributeError()

    @property
    def number_group(self) -> str:
        return self._number_group

    @number_group.setter
    def number_group(self, number_group: str) -> None:
        if self._number_group is None:
            self._number_group = number_group
        else:
            raise AttributeError()

    @property
    def graph(self) -> nx.DiGraph:
        if self._graph is None:
            self.__create_graph()
        return self._graph

    @property
    def S(self) -> int:
        return self._source

    @property
    def separating_set(self, vertex1='S', vertex2='T'):
        # nodes
        ff = ford_fulkerson_algorithm(self, vertex1, vertex2, image_create=False)
        res = []
        for path in ff[1]:
            for node in path:
                if node != vertex1 and node != vertex2:
                    res.append(node)
        return sorted(list(set(res)))

    @property
    def dividing_set(self, vertex1='S', vertex2='T'):
        # edges
        ff = ford_fulkerson_algorithm(self, vertex1, vertex2, image_create=False)
        res = []
        for path in ff[1]:
            for i in range(len(path) - 1):
                res.append((path[i], path[i + 1]))
        return sorted(set(res))

    @property
    def diam(self) -> int:
        # применим алгоритм Флойда – Уоршалла
        # https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm

        paths = floyd_warshall_algo(self.graph)
        temp = []
        for i in paths:
            temp.append(i[0])

        m = len(max(temp, key=len))

        ans = []
        for path in paths:
            if len(path[0]) == m:
                ans.append(path)
        return (m - 1, ans)

    @property
    def T(self) -> int:
        return self._destination

    @property
    def adj_matrix(self) -> list[list[int]]:
        length = len(self.graph.nodes)
        mat = [None] * length
        for el in self.graph.nodes:
            el_pos = self.renamer.correction_function(el)
            mat[el_pos] = [0] * length
            d = self.graph[self.renamer(el_pos)]
            for k in d:
                mat[el_pos][self.renamer.correction_function(k)] = d[k]['max_flow']
        return mat

    def save_graph_image(self, filename: str, path: list[str] | None = None, edges: list[str] = None, colored_edges = None) -> None:
        pos = circular_layout_but_better(self.renamer.sort_elements(self.graph))
        plt.figure(figsize=(10, 10))



        if edges == None:
            nx.draw(self.graph, pos=pos, with_labels=True,
                font_weight='bold', arrows=True)
        else:
            nx.draw(self.graph, pos=pos, with_labels=True,
                font_weight='bold', arrows=True, edgelist=edges)

        if path != None:
            nx.draw_networkx_edges(self.graph, pos, edgelist=[(path[i], path[i + 1]) for i in range(len(path) - 1)], edge_color='r', width=2)

        if edges == None:
            edge_labels = {(u, v): f"{d['current_flow']}/{d['max_flow']}" for u, v, d in self.graph.edges(data=True)}
            nx.draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=edge_labels)
        else:
            edge_labels = {}
            for u, v, d in self.graph.edges(data=True):
                if (u, v) in edges:
                    edge_labels[(u, v)] = f"{d['current_flow']}/{d['max_flow']}"
            nx.draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=edge_labels)

        if colored_edges != None:
            nx.draw_networkx_edges(self.graph, pos, edgelist=colored_edges, edge_color='r', width=1, style='dashed')



        plt.savefig(filename, dpi=300)
        plt.close()

    def _get_path_in_ford_fulkerson_algorithm(self, s: int, t: int, parent: list[int]) -> list[str]:
        path = []
        i = t
        while i != s:
            path.append(self.renamer(i))
            i = parent[i]
        path.append(self.renamer(s))
        return path[::-1]

    def _update_graph_in_ford_fulkerson_algorithm(self, vertex1: int | str, vertex2: int | str, value: int):

        self.graph[self.renamer(vertex1)][self.renamer(vertex2)]["current_flow"] += value

    def __create_graph(self) -> None:
        self._graph = nx.DiGraph()
        N = 7 + self.var % 4
        flag = 0
        if self.var % N == 1:
            flag = 1
        if self.var % N == N - 1:
            flag = 2
        if self.var % N == 0:
            flag = 3

        S = self.var % N
        T = (self.var + get_last_digit(self.number_group)) % N

        self._source = S
        self._destination = T
        self._renamer.add_elements_bundle(str(S), 'S')
        self._renamer.add_elements_bundle(str(T), 'T')

        for j in range(N):

            self._graph.add_node(self._renamer(j))
            self._graph.add_edge(self._renamer(j), self._renamer((j + 1) % N), current_flow=0, max_flow=j + 2)

            if flag == 1 or flag == 2:
                k_ = (j + 3) % N
            if flag == 3:
                if i % 2 == 0:
                    k_ = (j - 2) % N
                else:
                    k_ = (j + 2) % N
            if flag == 0:
                k_ = (j + self.var) % N
            self._graph.add_edge(self._renamer(j), self._renamer(k_), current_flow=0, max_flow=j + 2)


def circular_layout_but_better(G: dict) -> dict[str, np.float32]:
    theta = np.linspace(0, 1, len(G) + 1)[:0:-1] * 2 * np.pi + np.pi / 2
    theta = theta.astype(np.float32)
    pos = np.column_stack(
        [np.cos(theta), np.sin(theta), np.zeros((len(G), 0))]
    )
    pos = nx.rescale_layout(pos)
    pos = dict(zip(G, pos))

    return pos


if __name__ == "__main__":
    o = First_lab()
    o.var = 6
    o.number_group = "1042"

    #o.draw_cut([('0', '1')])
    #o.create_report()
    o.cut_creator()