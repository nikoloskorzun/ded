import networkx as nx

import matplotlib.pyplot as plt
import numpy

import numpy as np
def get_last_digit(line: str)->int:
    for c in line[::-1]:
        if c.isnumeric():
            return int(c)
    raise AttributeError("in str must be a digit")

class Renamer:
    def __init__(self):
        self._replace: dict[str, str] = {}
        self._invert: dict[str, str] = {}

    def add_elements_bundle(self, source: str, replacer: str) -> None:
        self._replace[source] = replacer
        self._invert[replacer] = source


    def _correction_function(self, el: str):
        if el in self._invert.keys():
            return int(self._invert[el])
        return int(el)

    def __call__(self, el: int):
        if str(el) in self._replace.keys():
            return self._replace[str(el)]
        return str(el)

    def sort_elements(self, graph: nx.DiGraph) -> list[str]:
        return sorted(graph, key=self._correction_function)
class First_lab:
    def __init__(self):
        self._var = None
        self._number_group = None
        self._graph = None
        self._source = None
        self._destination = None
        self._renamer = Renamer()


    @property
    def renamer(self):
        return self._renamer

    @property
    def var(self) -> int:
        return self._var

    @var.setter
    def var(self, var: int):
        if self._var is None:
            self._var = var
        else:
            raise AttributeError()

    @property
    def number_group(self) -> str:
        return self._number_group

    @number_group.setter
    def number_group(self, number_group: str):
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
    def S(self):
        return self._source
    @property
    def T(self):
        return self._destination


    def draw_(self):
        pos = circular_layout_but_better(self.renamer.sort_elements(self.graph))
        nx.draw(self.graph, pos=pos, with_labels=True,
                font_weight='bold', arrows=True)
        edge_labels = {(u, v): f"{d['current_flow']}/{d['max_flow']}" for u, v, d in self.graph.edges(data=True)}
        nx.draw_networkx_edge_labels(self.graph, pos=pos, edge_labels=edge_labels)
        #plt.show()
        plt.savefig('test.png')
        #ford_fulkerson_algorithm()
    def ford_fulkerson_algorithm(self):

        class Label:
            _states = {0: "not labeled", 1: "unviewed", 2: "labeled and viewed"}
            _state: int
            def __init__(self):
                self._state = 0
                self._number = None

                self._vertice = None
                self._sign = None

            @property
            def state(self):
                return self._state
            @state.setter
            def state(self, value: int):
                self._state = value
            @property
            def number(self):
                return self._number
            @number.setter
            def number(self, value: int):
                self._number = value
            @property
            def vertice(self):
                return self._vertice
            @vertice.setter
            def vertice(self, value: int):
                self._vertice = value



            def __str__(self):
                if self._state == 0:
                    return "not labeled"
                if self._state == 1:
                    return f"{self._vertice}{self._sign}; {self.number} unviewed"
                if self._state == 2:
                    return f"{self._vertice}{self._sign}; {self.number} viewed"


        N = len(self.graph)
        labels: list[Label] = []
        for _ in range(N):
            labels.append(Label())

        def stopping_condition()->bool:
            """
            Расстановка пометок по узлам, которые являются соседними для помеченных и не просмотренных узлов,
            продолжается до тех пор, пока либо узел NT окажется помеченным, либо нельзя будет больше пометить ни один узел
            и сток NT останется непомеченным.
            Если NT не может быть помечен, то не существует пути, увеличивающего поток, и, следовательно,
            полученный поток максимален и алгоритм заканчивает свою работу.
            Если же NT помечен, то на шаге 2 можно найти путь, увеличивающий поток

            """

            if labels[self.T].state != 0:
                return False


        def first_step():
            labels[self.S].vertice = 'S'
            labels[self.S].state = 1


            while stopping_condition():
                pass




        print(labels[1]())



    def __create_graph(self):
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
        self._renamer.add_elements_bundle(str(S), 'S')
        self._renamer.add_elements_bundle(str(T), 'T')



        for j in range(N):

            self._graph.add_node(self._renamer(j))
            self._graph.add_edge(self._renamer(j), self._renamer((j + 1) % N), current_flow = 0, max_flow = j+2)


            if flag == 1 or flag == 2:
                k_ = (j + 3) % N
            if flag == 3:
                if i % 2 == 0:
                    k_ = (j - 2) % N
                else:
                    k_ = (j + 2) % N
            if flag == 0:
                k_ = (j + self.var) % N
            self._graph.add_edge(self._renamer(j), self._renamer(k_), current_flow = 0, max_flow = j+2)

def circular_layout_but_better(G: dict):

    theta = np.linspace(0, 1, len(G)+1)[:0:-1] * 2 * np.pi + np.pi/2
    theta = theta.astype(np.float32)
    pos = np.column_stack(
        [np.cos(theta), np.sin(theta), np.zeros((len(G), 0))]
    )
    pos = nx.rescale_layout(pos)
    pos = dict(zip(G, pos))

    return pos

if __name__ == "__main__":
    o = First_lab()
    o.var = 9
    o.number_group = "1042"

    o.draw_()
    o.ford_fulkerson_algorithm()
