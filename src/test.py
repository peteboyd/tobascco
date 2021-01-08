import _nloptimize as nl
import numpy as np
import timeit
import math


class PCU_RUN(object):
    def __init__(self):
        self.ndim = 3
        self.diag_ind = 0
        self.init_x = np.array(
            [
                6.0,
                6.0,
                6.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )
        self.upper = np.array(
            [
                6.6,
                6.6,
                6.6,
                43.56,
                43.56,
                43.56,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ]
        )
        self.lower = np.array(
            [
                0.66,
                0.66,
                0.66,
                -43.56,
                -43.56,
                -43.56,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
                -0.5,
            ]
        )
        self.ip_mat = np.array(
            [
                [
                    0.992742059304,
                    -0.000455992035646,
                    -0.000203598724115,
                    0.999996216098,
                    0.0,
                    0.0,
                    0.999999996692,
                    0.999999967629,
                    -0.000927861442582,
                    0.0,
                    0.0,
                    7.21751427347e-05,
                    -0.000259709567379,
                    -0.000962483029883,
                    0.0,
                    0.0,
                    -0.000962483029883,
                    -0.000294391916385,
                ],
                [
                    -0.000455992035646,
                    0.9927944094,
                    0.000109808625679,
                    0.00147246358864,
                    0.0,
                    0.0,
                    -0.000528179344242,
                    -0.000313957709127,
                    0.999999011407,
                    0.0,
                    0.0,
                    0.99999980783,
                    0.999999977854,
                    0.000325131550703,
                    0.0,
                    0.0,
                    0.000325131550703,
                    -3.31857348392e-05,
                ],
                [
                    -0.000203598724115,
                    0.000109808625679,
                    0.992725316569,
                    0.00175846332081,
                    0.0,
                    0.0,
                    -0.000241099696477,
                    7.52815424533e-06,
                    -0.00121467990996,
                    0.0,
                    0.0,
                    -0.000214922276496,
                    3.38381010818e-05,
                    0.99999968894,
                    0.0,
                    0.0,
                    0.99999968894,
                    0.999999985649,
                ],
                [
                    0.999996216098,
                    0.00147246358864,
                    0.00175846332081,
                    0.18410773455,
                    0.999999859312,
                    0.999999905638,
                    0.999999967124,
                    0.999996871803,
                    0.000997995010029,
                    0.0,
                    0.0,
                    0.00199999200005,
                    0.00166859647972,
                    0.000999995500026,
                    0.0,
                    0.0,
                    0.000999995500026,
                    0.00166739460053,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.999999859312,
                    0.16034894123,
                    0.999999977285,
                    0.999999903151,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.999999905638,
                    0.999999977285,
                    0.160324555335,
                    0.999999894914,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.999999996692,
                    -0.000528179344242,
                    -0.000241099696477,
                    0.999999967124,
                    0.999999903151,
                    0.999999894914,
                    0.18410773455,
                    0.999999946152,
                    -0.000999999000002,
                    0.0,
                    0.0,
                    0.0,
                    -0.000331894034618,
                    -0.0009999995,
                    0.0,
                    0.0,
                    -0.0009999995,
                    -0.000331882562435,
                ],
                [
                    0.999999967629,
                    -0.000313957709127,
                    7.52815424533e-06,
                    0.999996871803,
                    0.0,
                    0.0,
                    0.999999946152,
                    0.999982968236,
                    -0.000786106906513,
                    0.0,
                    0.0,
                    0.000214140927407,
                    -0.000117691267447,
                    -0.000751325696124,
                    0.0,
                    0.0,
                    -0.000751325696124,
                    -8.32853546949e-05,
                ],
                [
                    -0.000927861442582,
                    0.999999011407,
                    -0.00121467990996,
                    0.000997995010029,
                    0.0,
                    0.0,
                    -0.000999999000002,
                    -0.000786106906513,
                    0.18410773455,
                    0.999999894914,
                    0.999999903151,
                    0.999999967124,
                    0.999998997217,
                    -0.000998998501502,
                    0.0,
                    0.0,
                    -0.000998998501502,
                    -0.00135763123306,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.999999894914,
                    0.160324555335,
                    0.999999977285,
                    0.999999905638,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.999999903151,
                    0.999999977285,
                    0.16034894123,
                    0.999999859312,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    7.21751427347e-05,
                    0.99999980783,
                    -0.000214922276496,
                    0.00199999200005,
                    0.0,
                    0.0,
                    0.0,
                    0.000214140927407,
                    0.999999967124,
                    0.999999905638,
                    0.999999859312,
                    0.18410773455,
                    0.999999914002,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.000357964592394,
                ],
                [
                    -0.000259709567379,
                    0.999999977854,
                    3.38381010818e-05,
                    0.00166859647972,
                    0.0,
                    0.0,
                    -0.000331894034618,
                    -0.000117691267447,
                    0.999998997217,
                    0.0,
                    0.0,
                    0.999999914002,
                    0.999930869594,
                    0.000249012122044,
                    0.0,
                    0.0,
                    0.000249012122044,
                    -0.000109174088881,
                ],
                [
                    -0.000962483029883,
                    0.000325131550703,
                    0.99999968894,
                    0.000999995500026,
                    0.0,
                    0.0,
                    -0.0009999995,
                    -0.000751325696124,
                    -0.000998998501502,
                    0.0,
                    0.0,
                    0.0,
                    0.000249012122044,
                    0.18410773455,
                    0.999999894914,
                    0.999999903151,
                    0.999999967124,
                    0.99999971274,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.999999894914,
                    0.160324555335,
                    0.999999977285,
                    0.999999905638,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.999999903151,
                    0.999999977285,
                    0.16034894123,
                    0.999999859312,
                    0.0,
                ],
                [
                    -0.000962483029883,
                    0.000325131550703,
                    0.99999968894,
                    0.000999995500026,
                    0.0,
                    0.0,
                    -0.0009999995,
                    -0.000751325696124,
                    -0.000998998501502,
                    0.0,
                    0.0,
                    0.0,
                    0.000249012122044,
                    0.999999967124,
                    0.999999905638,
                    0.999999859312,
                    0.18410773455,
                    0.99999971274,
                ],
                [
                    -0.000294391916385,
                    -3.31857348392e-05,
                    0.999999985649,
                    0.00166739460053,
                    0.0,
                    0.0,
                    -0.000331882562435,
                    -8.32853546949e-05,
                    -0.00135763123306,
                    0.0,
                    0.0,
                    -0.000357964592394,
                    -0.000109174088881,
                    0.99999971274,
                    0.0,
                    0.0,
                    0.99999971274,
                    1.0,
                ],
            ]
        )
        self.nz = [
            (1, 1),
            (1, 7),
            (1, 13),
            (1, 16),
            (1, 8),
            (1, 3),
            (1, 11),
            (1, 2),
            (1, 6),
            (1, 12),
            (1, 0),
            (1, 17),
            (7, 7),
            (7, 13),
            (7, 16),
            (7, 8),
            (7, 3),
            (7, 11),
            (7, 2),
            (7, 6),
            (7, 12),
            (7, 0),
            (7, 17),
            (13, 13),
            (13, 16),
            (13, 8),
            (13, 3),
            (13, 11),
            (13, 2),
            (13, 6),
            (13, 12),
            (13, 0),
            (13, 17),
            (16, 16),
            (16, 8),
            (16, 3),
            (16, 11),
            (16, 2),
            (16, 6),
            (16, 12),
            (16, 0),
            (16, 17),
            (8, 8),
            (8, 3),
            (8, 11),
            (8, 2),
            (8, 6),
            (8, 12),
            (8, 0),
            (8, 17),
            (3, 3),
            (3, 11),
            (3, 2),
            (3, 6),
            (3, 12),
            (3, 0),
            (3, 17),
            (11, 11),
            (11, 2),
            (11, 6),
            (11, 12),
            (11, 0),
            (11, 17),
            (2, 2),
            (2, 6),
            (2, 12),
            (2, 0),
            (2, 17),
            (6, 6),
            (6, 12),
            (6, 0),
            (6, 17),
            (12, 12),
            (12, 0),
            (12, 17),
            (0, 0),
            (0, 17),
            (17, 17),
            (4, 4),
            (4, 5),
            (4, 6),
            (4, 3),
            (5, 5),
            (5, 6),
            (5, 3),
            (6, 6),
            (6, 3),
            (3, 3),
            (10, 10),
            (10, 11),
            (10, 8),
            (10, 9),
            (11, 11),
            (11, 8),
            (11, 9),
            (8, 8),
            (8, 9),
            (9, 9),
            (13, 13),
            (13, 14),
            (13, 15),
            (13, 16),
            (14, 14),
            (14, 15),
            (14, 16),
            (15, 15),
            (15, 16),
            (16, 16),
        ]
        self.cycle_rep = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.cycle_cocycle_I = np.array(
            [
                [
                    -0.166666666667,
                    1.38777878078e-17,
                    5.21188031005e-17,
                    5.55111512313e-17,
                    -0.833333333333,
                    -0.666666666667,
                    -0.5,
                    -0.333333333333,
                    -0.166666666667,
                    4.16333634234e-17,
                    2.77555756156e-17,
                    1.42233772503e-17,
                    9.71377383911e-18,
                    1.38777878078e-17,
                    3.3923481308e-18,
                    2.15876699233e-18,
                    1.50342701251e-18,
                    4.62592926927e-19,
                ],
                [
                    0.0,
                    -0.166666666667,
                    1.38777878078e-17,
                    2.77555756156e-17,
                    2.77555756156e-17,
                    2.77555756156e-17,
                    2.77555756156e-17,
                    2.77555756156e-17,
                    2.77555756156e-17,
                    -0.833333333333,
                    -0.666666666667,
                    -0.5,
                    -0.333333333333,
                    -0.166666666667,
                    1.38777878078e-17,
                    1.32611478222e-17,
                    4.85858098545e-18,
                    -2.77487993521e-18,
                ],
                [
                    0.0,
                    0.0,
                    -0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    -0.666666666667,
                    -0.5,
                    -0.333333333333,
                    -0.166666666667,
                ],
                [
                    -0.166666666667,
                    -3.46944695195e-18,
                    -3.3923481308e-18,
                    0.0,
                    0.166666666667,
                    -0.666666666667,
                    -0.5,
                    -0.333333333333,
                    -0.166666666667,
                    3.46944695195e-18,
                    6.93889390391e-18,
                    4.16062583691e-18,
                    5.54975987041e-18,
                    1.38777878078e-17,
                    3.3923481308e-18,
                    2.15876699233e-18,
                    1.50342701251e-18,
                    4.62592926927e-19,
                ],
                [
                    -0.166666666667,
                    0.0,
                    2.43632274848e-17,
                    2.77555756156e-17,
                    0.166666666667,
                    0.333333333333,
                    -0.5,
                    -0.333333333333,
                    -0.166666666667,
                    2.77555756156e-17,
                    2.77555756156e-17,
                    1.42233772503e-17,
                    9.71377383911e-18,
                    1.38777878078e-17,
                    3.3923481308e-18,
                    2.15876699233e-18,
                    1.50342701251e-18,
                    4.62592926927e-19,
                ],
                [
                    -0.166666666667,
                    6.93889390391e-18,
                    -3.3923481308e-18,
                    0.0,
                    0.166666666667,
                    0.333333333333,
                    0.5,
                    -0.333333333333,
                    -0.166666666667,
                    -6.93889390391e-18,
                    -1.38777878078e-17,
                    -7.98243849492e-18,
                    -1.3891340335e-18,
                    1.38777878078e-17,
                    3.3923481308e-18,
                    2.15876699233e-18,
                    1.50342701251e-18,
                    4.62592926927e-19,
                ],
                [
                    -0.166666666667,
                    0.0,
                    -3.3923481308e-18,
                    0.0,
                    0.166666666667,
                    0.333333333333,
                    0.5,
                    0.666666666667,
                    -0.166666666667,
                    0.0,
                    0.0,
                    4.16062583691e-18,
                    5.54975987041e-18,
                    1.38777878078e-17,
                    3.3923481308e-18,
                    2.15876699233e-18,
                    1.50342701251e-18,
                    4.62592926927e-19,
                ],
                [
                    -0.166666666667,
                    0.0,
                    -9.40605618085e-17,
                    -1.11022302463e-16,
                    0.166666666667,
                    0.333333333333,
                    0.5,
                    0.666666666667,
                    0.833333333333,
                    -1.11022302463e-16,
                    -1.11022302463e-16,
                    -5.27464356914e-17,
                    -3.33121117496e-17,
                    -6.93889390391e-17,
                    -1.6961740654e-17,
                    -1.07938349616e-17,
                    -7.51713506257e-18,
                    -2.31296463464e-18,
                ],
                [
                    0.0,
                    -0.166666666667,
                    1.38777878078e-17,
                    2.77555756156e-17,
                    2.77555756156e-17,
                    2.77555756156e-17,
                    2.77555756156e-17,
                    2.77555756156e-17,
                    2.77555756156e-17,
                    0.166666666667,
                    -0.666666666667,
                    -0.5,
                    -0.333333333333,
                    -0.166666666667,
                    1.38777878078e-17,
                    1.32611478222e-17,
                    4.85858098545e-18,
                    -2.77487993521e-18,
                ],
                [
                    0.0,
                    -0.166666666667,
                    -1.38777878078e-17,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.166666666667,
                    0.333333333333,
                    -0.5,
                    -0.333333333333,
                    -0.166666666667,
                    1.38777878078e-17,
                    1.78859477142e-17,
                    1.17974748894e-17,
                    -2.77487993521e-18,
                ],
                [
                    0.0,
                    -0.166666666667,
                    -2.77555756156e-17,
                    -2.77555756156e-17,
                    -2.77555756156e-17,
                    -2.77555756156e-17,
                    -2.77555756156e-17,
                    -2.77555756156e-17,
                    -2.77555756156e-17,
                    0.166666666667,
                    0.333333333333,
                    0.5,
                    -0.333333333333,
                    -0.166666666667,
                    0.0,
                    -5.7056139327e-18,
                    6.9117888496e-19,
                    -8.3280279374e-18,
                ],
                [
                    0.0,
                    -0.166666666667,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.166666666667,
                    0.333333333333,
                    0.5,
                    0.666666666667,
                    -0.166666666667,
                    0.0,
                    1.2332799712e-18,
                    6.9117888496e-19,
                    -8.3280279374e-18,
                ],
                [
                    0.0,
                    -0.166666666667,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.166666666667,
                    0.333333333333,
                    0.5,
                    0.666666666667,
                    0.833333333333,
                    0.0,
                    -3.0222135558e-17,
                    -2.21990394816e-17,
                    4.44116314904e-17,
                ],
                [
                    0.0,
                    0.0,
                    -0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.333333333333,
                    -0.5,
                    -0.333333333333,
                    -0.166666666667,
                ],
                [
                    0.0,
                    0.0,
                    -0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.333333333333,
                    0.5,
                    -0.333333333333,
                    -0.166666666667,
                ],
                [
                    0.0,
                    0.0,
                    -0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.333333333333,
                    0.5,
                    0.666666666667,
                    -0.166666666667,
                ],
                [
                    0.0,
                    0.0,
                    -0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.166666666667,
                    0.333333333333,
                    0.5,
                    0.666666666667,
                    0.833333333333,
                ],
                [
                    0.0,
                    0.0,
                    -0.166666666667,
                    -0.833333333333,
                    -0.833333333333,
                    -0.833333333333,
                    -0.833333333333,
                    -0.833333333333,
                    -0.833333333333,
                    -0.833333333333,
                    -0.833333333333,
                    -0.833333333333,
                    -0.833333333333,
                    -0.833333333333,
                    -0.666666666667,
                    -0.5,
                    -0.333333333333,
                    -0.166666666667,
                ],
            ]
        )

    def optim_call(self):
        nzi = np.array([i[0] for i in self.nz])
        nzj = np.array([i[1] for i in self.nz])
        print(type(self.upper[0]))
        x = nl.nloptimize(
            self.ndim,
            self.diag_ind,
            self.lower,
            self.upper,
            self.init_x,
            self.cycle_rep,
            self.cycle_cocycle_I,
            self.ip_mat,
            nzi,
            nzj,
        )
        print(x)

    def init_min_function_nlopt(self):
        f = math.factorial
        ndim = self.ndim
        cocycle_size = self.cycle_cocycle_I.shape[0] - self.cycle_rep.shape[0]
        angle_inds = f(self.ndim) / f(2) / f(self.ndim - 2)
        nzi, nzj = [], []
        for (i, j) in self.nz:
            nzi.append(i)
            nzj.append(j)
        nz = [np.array(nzi), np.array(nzj)]
        iu = np.triu_indices(self.ndim, k=1)
        il = np.tril_indices(self.ndim, k=-1)
        scale_ind = self.diag_ind

        def min_function_nlopt(x, grad):
            """TODO - fix this so it works.
            the metric tensor needs to be squared in the diagonal
            and the proper dot product needs to be represented in 
            the off-diagonals.

            the cocycle_rep needs to be properly concatenated with
            the cycle_rep.
            """
            if grad.size > 0:
                grad[:] = 0.0
            # decompress 'x' into useable forms
            # mt, cocycle_rep = self.convert_params(x, ndim, angle_inds, cocycle_size, iu, il)
            cell_lengths = x[:ndim]
            angles = x[ndim : ndim + angle_inds]
            cocycle = x[ndim + angle_inds :]
            mt = np.empty((ndim, ndim))
            # convention alpha --> b,c beta --> a,c gamma --> a,b
            # in the metric tensor, these are related to the
            # (1,2), (0,2), and (0,1) array elements, which
            # are in the reversed order of how they would
            # be iterated.
            # assuming the parameters are defined in 'x' as
            # x[3] --> a.b  \
            # x[4] --> a.c  |--> these are in reversed order.
            # x[5] --> b.c  /
            # rev_angles = angles[::-1]
            mt[iu] = angles
            mt[il] = angles
            # obtain diagonal and off-diagonal elements of the metric tensor
            mt[np.diag_indices_from(mt)] = cell_lengths
            # for i in range(ndim):
            #    mt[i,i] = cell_lengths[i] * cell_lengths[i]
            # for (i,j),(k,l) in zip(zip(*iu), zip(*il)):
            #    mt[i,j] = mt[i,j] * cell_lengths[i] * cell_lengths[j]
            #    mt[k,l] = mt[k,l] * cell_lengths[k] * cell_lengths[l]
            cocycle_rep = np.reshape(cocycle, (cocycle_size, ndim))
            # obtain net embedding defined by these parameters.
            rep = np.concatenate((self.cycle_rep[:], cocycle_rep[:]))
            la = np.dot(self.cycle_cocycle_I, rep)
            M = np.dot(np.dot(la, mt), la.T)
            scale_fact = M[scale_ind, scale_ind]  # .max()
            for (i, j) in zip(*np.triu_indices_from(M)):
                val = M[i, j]
                if i != j:
                    v = val / np.sqrt(M[i, i]) / np.sqrt(M[j, j])
                    M[i, j] = v
                    M[j, i] = v
            M[np.diag_indices_from(M)] /= scale_fact
            # length_part = np.diag(M)
            # nz_triu = np.nonzero(np.triu(matching_ip_matrix,k=1))
            # angle_part = M[nz_triu]
            sol = (M[nz] - self.ip_mat[nz]) ** 2
            ret_val = np.sum(sol)
            # ret_val = np.random.random()*5000.
            # print 'length diff %15.9f'%np.sum(np.abs(length_part - np.diag(matching_ip_matrix)))
            # print 'angle diff  %15.9f'%np.sum(np.abs(angle_part - matching_ip_matrix[nz_triu]))
            # print 'functn val  %15.9f'%ret_val
            # print M[nz] - matching_ip_matrix[nz]
            # print ret_val
            return ret_val

        return min_function_nlopt


pcu = PCU_RUN()
for i in range(1):
    pcu.optim_call()
    # f = pcu.init_min_function_nlopt()
    # print f(pcu.init_x, np.array([]))
    # c = np.concatenate((pcu.cycle_rep, np.zeros(((pcu.init_x.shape[0]-6)/3, 3))))
    # print pcu.cycle_cocycle_I.shape, c.shape
    # print np.dot(pcu.cycle_cocycle_I, c)
