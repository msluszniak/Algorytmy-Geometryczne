import numpy as np
from enum import Enum


class EventType(Enum):
    site = 0
    circle = 1

    def __lt__(self, other):
        #priorytet mają zdarzenia punktowe przed okręgowymi
        return self.value < other.value


class Event:
    epsilon = 1e-9

    def __init__(self, y, event_type, site=None, arc=None, point=None):
        '''
        deklaracja eventu i jego parametrów
        :param y: współrzędna y eventu, po tym jest sortowany event w strukturze zdarzeń
        :param event_type: rodzaj eventu, punktowy albo okręgowy
        :param site: związany z eventem obszar
        :param arc: łuk związany z eventem
        :param point: punkt związany z eventem
        '''
        self.y = y
        self.site = site
        self.type = event_type
        self.arc = arc
        self.point = point

    def __lt__(self, other):
        #jest też zdefiniowane porównywanie pomiędzy obiektami typu event, żeby nie robić tego w priority queue
        return self.y > other.y or (np.abs(self.y - other.y) < self.epsilon and self.type < other.type)
    def __hash__(self):
        return hash(tuple(self.point))