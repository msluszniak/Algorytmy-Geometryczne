import numpy as np
from Event import Event, EventType
from Beach_line import Arc, BeachLine
from Metric import euclidean_2d
from Voronoi_diagram import VoronoiDiagram
from queue import PriorityQueue
import Visualization as Vis


def diagram_edges(diagram):
    '''

    :param diagram: diagram czyli parametr FortuneAlgorith
    :return: edges; lista krawędzi w postaci tupli dwóch punktów
    '''
    edges = []
    for e in diagram.half_edges:
        if e.origin is not None and e.destination is not None:
            edges.append((e.origin.point, e.destination.point))
    return edges


class FortuneAlgorithm:
    metrics = {'euclidean_2d': euclidean_2d}

    def __init__(self, points, named_metric='euclidean_2d', metric=None):
        '''
        inicjalizuje podstawowe struktury algorytmu zamiatającego, linie brzegową oraz metryke w jakiej wyznaczamy diagram, a także sam pusty diagram
        :param points: lista punktów dla których wyznaczamy diagram Voronoi
        :param named_metric: Wybrana metryka, co prawda mamy zrobioną tylko metrykę euklidesową 2d, ale zostawiamy furtkę na inne metryki
        :param metric: wybrana metryka
        '''

        self.diagram = VoronoiDiagram(points)

        if metric is not None:
            self.metric = metric
            self.compute_convergence_point = metric.compute_convergence_point
        elif named_metric in FortuneAlgorithm.metrics:
            self.metric = FortuneAlgorithm.metrics[named_metric]
            self.compute_convergence_point = metric.compute_convergence_point
        else:
            raise ValueError('Can\'t resolve metric function', metric, 'please choose from list, \
            or implement your own. list: ', FortuneAlgorithm.metrics.keys())
        # linia brzegowa
        self.beach_line = BeachLine(self.metric.compute_breakpoint)

    def construct(self, points):
        """
        Wykonuje właściwe działanie algorytmu poprzez obsługę zdarzeń typu zdarzenie punktowe oraz zdarzenie kołowe.
        Oprócz tego obługuje tworzenie się scen wizualizacji w miarę obsługiwania kolejnych eventów.
        :return: Scenes; lista obiektów typu Scene z narzędzia graficznego
        """

        max_x = -float("inf")
        min_x = float("inf")
        for point in points:
            if point[0] > max_x:
                max_x = point[0]
            if point[0] < min_x:
                min_x = point[0]

        scenes = []
        events = PriorityQueue()

        for s in self.diagram.sites:
            event = Event(s.point[1], EventType.site, site=s, point=s.point)
            events.put(event)

        not_valid_events = set()

        while not events.empty():
            event = events.get()
            if event.type.value == 0:
                x_left, x_right = self.left_and_right_bound()
                edges = diagram_edges(self.diagram)
                parabolas = []
                arc_list = self.beach_line.inorder()
                for node in arc_list:
                    breakpoint_right = np.inf
                    breakpoint_left = -np.inf
                    if node.prev is not None:
                        breakpoint_left = self.metric.compute_breakpoint(node.prev.site.point, node.site.point,
                                                                         event.point[1])

                    if node.next is not None:
                        breakpoint_right = self.metric.compute_breakpoint(node.site.point, node.next.site.point,
                                                                          event.point[1])
                    p = (node.site.point[1] - event.point[1]) / 2
                    h = node.site.point[0]
                    k = event.point[1] + p
                    if p != 0:
                        a = 1 / (4 * p)
                        b = -(h / (2 * p))
                        c = (h ** 2) / (4 * p) + k

                        if breakpoint_left > x_left and breakpoint_right < x_right:
                            x = np.arange(breakpoint_left, breakpoint_right, 0.001)
                        elif breakpoint_left < x_left and breakpoint_right < x_right:
                            x = np.arange(x_left, breakpoint_right, 0.001)
                        elif breakpoint_left > x_left and breakpoint_right > x_right:
                            x = np.arange(breakpoint_left, x_right, 0.001)
                        else:
                            x = np.arange(x_left, x_right, 0.001)
                        y = a * (x ** 2) + b * x + c

                        points1 = list(zip(x, y))
                        parabolas.append(points1)

                result = []
                for parabola in parabolas:
                    for point in parabola:
                        result.append(point)

                scenes.append(Vis.Scene(
                    [Vis.PointsCollection([event.point], color='red'), Vis.PointsCollection(points, color='purple'),
                     Vis.PointsCollection(result, s=3, color='green')],
                    [Vis.LinesCollection([((min_x, event.point[1],), (max_x, event.point[1]))], color='brown'),
                     Vis.LinesCollection(edges, color='red')]
                    ))

            if event in not_valid_events:
                continue

            if event.type == EventType.site:
                self.handle_site_event(event, not_valid_events, events)
            else:
                self.handle_circle_event(event, not_valid_events, events)

        edges = diagram_edges(self.diagram)
        scenes.append(
            Vis.Scene([Vis.PointsCollection(points, color='purple')
                       ],
                      [
                          Vis.LinesCollection(edges, color='red')]
                      ))
        self.bound()
        edges = diagram_edges(self.diagram)
        scenes.append(
            Vis.Scene([Vis.PointsCollection(points, color='purple')
                       ],
                      [
                          Vis.LinesCollection(edges, color='red')]
                      ))
        return scenes

    def break_arc_by_site(self, arc, site):
        """
        :param arc: łuk, który ulegnie podziałowi na trzy mniejsze łuki
        :param site: punkt, który powoduje podział łuków na trzy
        :return: middle arc; jest to łuk związany z punktem wykrytym w ramach site eventu
        """
        middle_arc = Arc(site)

        left_arc = Arc(arc.site)
        left_arc.left_half_edge = arc.left_half_edge

        right_arc = Arc(arc.site)
        right_arc.right_half_edge = arc.right_half_edge

        self.beach_line.replace(arc, middle_arc)
        self.beach_line.insert_before(middle_arc, left_arc)
        self.beach_line.insert_after(middle_arc, right_arc)

        return middle_arc

    def add_edge(self, left, right):
        '''
        funkcja dodaje półprostą w momencie obsługiwania zdarzenia punktowego bądź też usuwania łuku
        :param left: jest to lewy łuk, wtedy półprosta z nim związana jest prawą półprostą
        :param right: jest to prawy łuk względem lewego, wówczas dodawana półprosta jest jego lewą półprostą
        '''
        left.right_half_edge = self.diagram.add_half_edge(left.site.face)
        right.left_half_edge = self.diagram.add_half_edge(right.site.face)


    def add_event(self, left, middle, right, events, beachline_y):
        '''
        funckja odpowiada za dodanie zdarzenia (zdarzenie kołowe) do struktury zdarzeń (kolejki priorytetowej).
        Zdarzenie nie może być fałszywym alarmem, więc sprawdzamy to w is_valid
        :param left: lewy łuk
        :param middle: środkowy łuk
        :param right: prawy łuk
        :param events: struktura zdarzeń (kolejka priorytetowa)
        :param beachline_y: linia brzegowa (drzewo czerowono czarne, wzbogacone)
        '''

        y, convergence_point = self.metric.compute_convergence_point(left.site.point, middle.site.point,
                                                                     right.site.point)

        is_below_broom = y <= beachline_y

        left_point_is_moving_right = left.site.point[1] < middle.site.point[1]
        right_point_is_moving_right = middle.site.point[1] < right.site.point[1]

        left_initial_x = left.site.point[0] if left_point_is_moving_right else middle.site.point[0]
        right_initial_x = middle.site.point[0] if right_point_is_moving_right else right.site.point[0]

        is_valid = ((left_point_is_moving_right and left_initial_x < convergence_point[0]) or
                    ((not left_point_is_moving_right) and left_initial_x > convergence_point[0])) and \
                   ((right_point_is_moving_right and right_initial_x < convergence_point[0]) or
                    ((not right_point_is_moving_right) and right_initial_x > convergence_point[0]))

        if is_valid and is_below_broom:
            event = Event(y, EventType.circle, point=convergence_point, arc=middle)
            middle.event = event
            events.put(event)

    def handle_site_event(self, event, not_valid_events, events):
        '''
        funckcja odpowiada za obsługę zdarzenia punktowego, napierw dzieli łuk na trzy łuki po napotakniu punktu
        Następnie zaś jeśli jest to możliwe, dodaje zdarzenie kołowe (muszą być trzy punkty związane z kolejnymi łukami obok siebie)
        :param event: aktualnie obsługiwane zdarzenie
        :param not_valid_events: lista fałszywych alarmów
        :param events: struktura zdarzeń
        '''
        site = event.site

        if self.beach_line.is_empty():
            self.beach_line.set_root(Arc(site))
            return

        arc_above = self.beach_line.get_arc_above(site.point, site.point[1])

        if arc_above.event is not None:
            not_valid_events.add(arc_above.event)

        middle_arc = self.break_arc_by_site(arc_above, site)
        left_arc = middle_arc.prev
        right_arc = middle_arc.next

        self.add_edge(left_arc, middle_arc)

        middle_arc.right_half_edge = middle_arc.left_half_edge
        right_arc.left_half_edge = left_arc.right_half_edge

        if left_arc.prev is not None:
            self.add_event(left_arc.prev, left_arc, middle_arc, events, site.point[1])

        if right_arc.next is not None:
            self.add_event(middle_arc, right_arc, right_arc.next, events, site.point[1])

    def remove_arc(self, arc, vertex):
        '''
        funkcja odpowiadająca za usunięcie łuku, który zapadł się właśnie do punktu diagramu voronoi
        Oprócz usunięcia łuku z linii brzegowej, to dodaje ona utworzoną dzięki temu półprostą, a następnie domyka ją do krawędzi
        :param arc: Łuk który właśnie się zapadł do punktu i który należy usunąć
        :param vertex: jest to punkt, do którego właśnie zapadł się łuk paraboli
        '''
        arc.prev.right_half_edge.origin = vertex
        arc.left_half_edge.destination = vertex

        arc.right_half_edge.origin = vertex
        arc.next.left_half_edge.destination = vertex

        self.beach_line.delete(arc)

        self.add_edge(arc.prev, arc.next)

        arc.prev.right_half_edge.destination = vertex
        arc.next.left_half_edge.origin = vertex


    def handle_circle_event(self, event, not_valid_events, events):
        '''
        Metoda odpowiedzialna za obsługę zdarzenia okręgowego , najpierw do diagramu Voronoia zostaje dodany nowo utworzony punkt
        Następnie z linii brzegowej zostaje usunięty łuk, który właśnie zapadł się do punktu
        Na końcu sprawdza czy może w łukach nie są trzymane fałszywe alarmy, jeśli tak to dodaje je do setu not_valid_events
        na samym końcu próbuje jak poprzednio dla zdarzeń punktowych dodać zdarzenia okręgowe
        :param event: zdarzenie aktualnie obsługiwane (jest ono zdarzeniem okręgowym)
        :param not_valid_events: zbiór fałszywych alarmów
        :param events: struktura zdarzeń
        '''
        point = event.point
        arc = event.arc

        voronoi_vertex = self.diagram.add_vertex(point)

        left_arc = arc.prev
        right_arc = arc.next

        if left_arc is not None and left_arc.event is not None:
            not_valid_events.add(left_arc.event)

        if right_arc is not None and right_arc.event is not None:
            not_valid_events.add(right_arc.event)

        self.remove_arc(arc, voronoi_vertex)

        if left_arc.prev is not None:
            self.add_event(left_arc.prev, left_arc, right_arc, events, event.y)

        if right_arc.next is not None:
            self.add_event(left_arc, right_arc, right_arc.next, events, event.y)

    def adjust_box(self, x_left, y_left, x_right, y_right, points):
        '''
        metoda ta dopasowuje rysowanie półprostych tak, żeby półproste rysowane na końcu, które odpowiadają za nieskończone obszary diagramu Voronoi
        przylegały do obramówki plota
        :param x_left: współrzędna x lewego dolnego rogu plota
        :param y_left: współrzędna y lewego dolnego rogu plota
        :param x_right: współrzędna x prawego górnego rogu plota
        :param y_right: współrzędna y prawego górnego rogu plota
        :param points: punkty przekazane do znajdowania diagramu Voronoia
        :return:
        '''
        for p in points:
            gap_between_box = np.array([0.5, 0.5])
            x_left = min(x_left, p.point[0] - gap_between_box[0])
            y_left = min(y_left, p.point[1] - gap_between_box[1])

            x_right = max(x_right, p.point[0] + gap_between_box[0])
            y_right = max(y_right, p.point[1] + gap_between_box[1])

        return x_left, y_left, x_right, y_right

    def get_intersection(self, x_left, y_left, x_right, y_right, origin, direction):
        '''
        funkcja jest wywoływana na końcu, gdy należy obługżyć zalegające jeszcze łuki w linii brzegowej po przejściu wszystkich zdarzeń ze struktury
        zdarzeń. wtedy wyznacza intersection które jest punktem przecięcia się półprostych odpowiedzialnych za nieskończone obszary diagramu Voronoi
        :param x_left: współrzędna x lewego dolnego rogu plota
        :param y_left: współrzędna y lewego dolnego rogu plota
        :param x_right: współrzędna x prawego górnego rogu plota
        :param y_right: współrzędna y prawego górnego rogu plota
        :param origin: punkt będący środkiem krawędzi łączącej początki półprostych, dla których szukamy punktu przecięcia
        :param direction: jest to wektor powstały poprzez odjęcie współrzędej jednego z początków i drugiego z początków. Następnie współrzędne
        x i y są zamienione , zaś współrzędna x ( już po zamianie_ jest mnożona przez -1. Jest to robione dla uproszczenia potem obliczeń
        Same obliczenia do wyznaczania punktu przecięcia wynikają z geometrii analitycznej , zaczerpneliśmy je z https://github.com/pvigier/FortuneAlgorithm/blob/master/src/Beachline.cpp
        :return:
        '''
        intersection = None

        t1, t2 = None, None

        eps = 1e-9

        if direction[0] > eps:
            t1 = (x_right - origin[0]) / direction[0]
            intersection = origin + t1 * direction
        elif direction[0] < -eps:
            t1 = (x_left - origin[0]) / direction[0]
            intersection = origin + t1 * direction

        if direction[1] > eps:
            t2 = (y_right - origin[1]) / direction[1]

            if t2 < t1:
                intersection = origin + t2 * direction

        elif direction[1] < -eps:
            t2 = (y_left - origin[1]) / direction[1]

            if t2 < t1:
                intersection = origin + t2 * direction

        return intersection

    def bound(self, x_left=float("inf"), y_left=float("inf"), x_right=-float("inf"), y_right=-float("inf")):
        '''
        funkcja wywoływana na końcu constructa. Odpowiada ona za dodanie odcinków symbolizujących nieskończone półproste diagramu voronoia
        :param x_left: współrzędna x lewego dolnego rogu plota
        :param y_left: współrzędna y lewego dolnego rogu plota
        :param x_right: współrzędna x prawego górnego rogu plota
        :param y_right: współrzędna y prawego górnego rogu plota
        '''
        x_left, y_left, x_right, y_right = self.adjust_box(x_left, y_left, x_right, y_right, self.diagram.sites)
        x_left, y_left, x_right, y_right = self.adjust_box(x_left, y_left, x_right, y_right, self.diagram.vertices)

        if not self.beach_line.is_empty():
            left_arc = self.beach_line.get_leftmost_arc()
            right_arc = left_arc.next

            while right_arc is not None:
                direction = (left_arc.site.point - right_arc.site.point)[[1, 0]]
                direction[0] *= -1
                origin = (left_arc.site.point + right_arc.site.point) * 0.5

                intersection = self.get_intersection(x_left, y_left, x_right, y_right, origin, direction)
                vertex = self.diagram.add_vertex(intersection)

                left_arc.right_half_edge.origin = vertex
                right_arc.left_half_edge.destination = vertex

                left_arc = right_arc
                right_arc = right_arc.next

    def left_and_right_bound(self, x_left=float("inf"), y_left=float("inf"), x_right=-float("inf"),
                             y_right=-float("inf")):
        #funckja przydatna do określenia dokąd należy rysować parabole (żeby nie wystawały znacznie poza wykres)
        x_left, y_left, x_right, y_right = self.adjust_box(x_left, y_left, x_right, y_right, self.diagram.sites)
        x_left, y_left, x_right, y_right = self.adjust_box(x_left, y_left, x_right, y_right, self.diagram.vertices)
        return x_left, x_right
