from networkx import DiGraph
class Renamer:
    def __init__(self):
        self._replace: dict[str, str] = {}
        self._invert: dict[str, str] = {}

    def add_elements_bundle(self, source: str, replacer: str) -> None:
        self._replace[source] = replacer
        self._invert[replacer] = source

    def correction_function(self, el: str) -> int:
        if el in self._invert.keys():
            return int(self._invert[el])
        return int(el)

    def __call__(self, el: int) -> str:
        if str(el) in self._replace.keys():
            return self._replace[str(el)]
        return str(el)

    def sort_elements(self, graph: DiGraph) -> list[str]:
        return sorted(graph, key=self.correction_function)
