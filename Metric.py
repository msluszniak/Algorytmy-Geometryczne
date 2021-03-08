import numpy as np


class Metric:

    @staticmethod
    def compute_breakpoint(point1, point2, l):
        """
        :param point1: ognisko paraboli związanej z pierwszą parabolą
        :param point2: ognisko paraboli związanej z drugą parabolą
        :param l: współrzędna y kierownicy dla obu parabol
        :return: zwraca punkt przecięcia parabol (breakpoint)
        """
        pass

    @staticmethod
    def compute_convergence_point(point1, point2, point3):
        """
        Kiedy zachodzi zdarzenie okręgowe to należy wyznaczyć środek okręgu opisanego na tych trzech punktach oraz najmniejszą współrzędną y okręgu
        :param point1: pierwszy punkt
        :param point2: drugi punkt
        :param point3: trzeci punkt
        :return: zwraca punkt będący środkie okręgu opisanego na trzech punktach a także położenie y najniższego punktu okręgu
        """
        pass


class euclidean_2d(Metric):

    @staticmethod
    def compute_breakpoint(point1, point2, l):
        x1, y1 = point1
        x2, y2 = point2

        eps = 1e-9

        d1 = 0.5/((y1 - l) + eps)
        d2 = 0.5/((y2 - l) + eps)

        a = d1 - d2
        b = 2.0 * (d2*x2 - d1*x1)
        c = (x1*x1 - l*l + y1*y1) * d1 - (x2*x2 - l*l + y2*y2) * d2

        delta = b*b - 4.0*a*c

        return (-b + np.sqrt(np.abs(delta))) / (2.0*a + eps)

    @staticmethod
    def compute_convergence_point(point1, point2, point3):
        eps = 1e-9

        v1 = (point1-point2)[[1, 0]]
        v2 = (point2 - point3)[[1, 0]]
        v1[0] = -v1[0]
        v2[0] = -v2[0]

        delta = (point3 - point1) * 0.5
        t = (delta[0] * v2[1] - delta[1]*v2[0]) / (v1[0] * v2[1] - v1[1]*v2[0] + eps)

        center = 0.5 * (point1 + point2) + t*v1

        r = np.sqrt(np.sum((point1 - center)**2))

        y = center[1] - r

        return y, center