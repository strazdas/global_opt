#! coding: utf-8
'''
nD -> 1D problem partitioning by dividing longest simplex edge into two equal parts.
'''
from copy import copy
from itertools import permutations
from numpy import array as a, sqrt, hstack, vstack
from utils import enorm, l2norm, city_block_norm, show_pareto_front, show_partitioning,\
                  draw_bounds, remove_dominated_subinterval, show_lower_pareto_bound,\
                  draw_simplex_3d_euclidean_bounds, draw_two_objectives_for_2d_simplex,\
                  nm, draw_3d_objective_function, show_potential, wrap
import numpy as np
# from scipy.optimize import minimize
from numpy.linalg import det, inv
# from matplotlib import pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("error")


# def branin(X):
#     '''2D http://www.sfu.ca/~ssurjano/branin.html'''
#     x1 = X[0]
#     x2 = X[1]
# 
#     pi_ = np.pi # 3.15 # 41592653589793
# 
#     a_ = 1.
#     b = 5.1/(4. * pi_**2)
#     c = 5. / pi_
#     r = 6.
#     s = 10.
#     t = 1. / (8. * pi_)
#     value = a_*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1 - t) * np.cos(x1) + s
#     # print 'At point', X, 'value', value
#     return value

def rastrigin(X, A=1):
    '''https://en.wikipedia.org/wiki/Rastrigin_function'''
    n = len(X)
    return A*n + sum([x**2 - A * np.cos(2*np.pi*x) for x in X])

def rosenbrock(X):
    '''https://en.wikipedia.org/wiki/Rosenbrock_function'''
    return sum([100 * (X[i+1] - X[i]**2)**2 + (X[i]-1)**2 for i in range(len(X)-1)])

def hyperparabola(X):  # n->1
    '''Hyper-parabola with minimum at [1.]*n.'''
    return (a(X)-1).dot(a(X)-1)

def shubert(X):
    '''http://www.sfu.ca/~ssurjano/shubert.html'''
    sum1 = 0
    sum2 = 0
    for i in range(5):
        new1 = i * np.cos((i+1)*X[0]+i)
        new2 = i * np.cos((i+1)*X[1]+i)
        sum1 += new1
        sum2 += new2
    return sum1 * sum2


def triangulate(lb, ub):   # n->n
    '''Triangulates the given hyperrectangle using combinatoric triangulation.'''
    # Kombinatorinis hyperkūbo dalinimo metodas
    triangles = []
    for t in permutations(xrange(len(lb))):
        vertexes = [copy(lb)]
        triangles.append(vertexes)
        for i in xrange(len(lb)):
            vertexes.append([])
            for j in xrange(len(lb)):
                vertexes[-1].append(vertexes[-2][j])
            vertexes[-1][t[i]] = ub[t[i]]
    # todo: Hyperkūbo padalnimas su Hyperįstrižaine
    return triangles

def find_objective_values_for_vertexes(simplexes, f):
    '''Converts simplex list to unique vertex list and adds meta for each vertex.'''
    points = []
    for simplex in simplexes:
        for vertex in simplex:
            if vertex not in points:
                points.append(vertex)
            else:
                p_index = points.index(vertex)
                s_index = simplex.index(vertex)
                simplex.remove(vertex)
                simplex.insert(s_index, points[p_index])
    return [add_objective_values(p, f) for p in points]

def add_objective_values(x, f):
    '''Adds objective values to meta information for a single point (e.g. simplex vertex).'''
    if type(x) == list:
        x.append({'obj': (f(x),)})
        return x
    return [x, {'obj': (f(x),)}]

def find_meta_for_simplexes(simplexes):
    '''Groups points into intervals and computes tolerance for each interval.'''
    for simplex in simplexes:
        sort_vertexes_longest_edge_first(simplex)
        if type(simplex[-1]) == dict:
            simplex[-1]['value'] = find_simplex_value(simplex),
            simplex[-1]['size'] = find_simplex_size(simplex),
            # simplex[-1]['mins_AB'] = find_mins_AB(simplex, L)
            # simplex[-1]['approx_min_ABC'] = get_approx_lb_min(simplex, L)
            # simplex[-1]['tolerance'] = get_tolerance(simplex, L)
        else:
            simplex.append({
                'value':  find_simplex_value(simplex),
                'size': find_simplex_size(simplex),
                # 'mins_AB': find_mins_AB(simplex, L),
                # 'approx_min_ABC': get_approx_lb_min(simplex, L),
                # 'tolerance': get_tolerance(simplex, L),
            })
    return simplexes

def sort_vertexes_longest_edge_first(simplex):
    '''nD->nD Moves longest edge vertexes to the simplex vertex list beginning.'''
    # Find simplex edges lengths
    edge_lengths = []   # [(vertex_index, vertex_index, edge_length),]
    for i, j in permutations(range(len(simplex[:-1])+1), 2):
        if j > i:
            edge_lengths.append((i, j, l2norm(simplex[i][:-1], simplex[j][:-1])))


    # Get longest edge vertexes ids
    le_i, le_j, le_length = max(edge_lengths, key=lambda x: x[-1])

    # Move longest edge vertexes to simplex vertex list beginning
    vi = simplex[le_i]
    vj = simplex[le_j]
    simplex.remove(vi)
    simplex.remove(vj)
    simplex.insert(0, vj)
    simplex.insert(0, vi)
    return simplex

def find_simplex_value(simplex):
    ## Three:
    # return np.mean([v[-1]['obj'] for v in simplex[:-1]])
    ## Two:
    return np.mean([v[-1]['obj'] for v in simplex[:2]])
    ## One vertex:
    # return min([v[-1]['obj'][0] for v in simplex])

def find_simplex_size(simplex):
    return enorm(a(simplex[0][:-1]) - a(simplex[1][:-1]))


# def find_mins_AB(simplex, L):
#     ''' nD->nD
#     Finds AB' and B'A intersection, where A, B are longest edge vertexes
#     t - triangle (simplex).
#     y - objective values for each vertex.
#     Returns lower Lipschitz bound minimum for the first edge (made from first
#     and second vetexes).
#     '''
#     dist = l2norm(nm(simplex[0]), nm(simplex[1]))
#     x1 = a((0, simplex[0][-1]['obj'][0]))
#     x2 = a((dist, simplex[0][-1]['obj'][0] - L*dist))
#     x3 = a((dist, simplex[1][-1]['obj'][0]))
#     x4 = a((0, simplex[1][-1]['obj'][0] - L*dist))
#
#     ## 2D line intersection based on  http://mathworld.wolfram.com/Line-LineIntersection.html
#     av = x2 - x1
#     bv = x4 - x3
#     cv = x3 - x1
#
#     s = x1 + av * (np.cross(cv, bv) * (np.cross(av, bv))/( enorm(np.cross(av, bv))**2 ))
#     X = a(simplex[0][:-1]) + s[0]/float(dist) * (a(simplex[1][:-1]) - a(simplex[0][:-1]))
#     return [list(X) + [s[1]]]



## Tolerance methods
def get_lower_bound_values_3D(simplex, point, L):
    '''Returns bouth objective values at a given point in simplex'''
    v1 = simplex[0]
    v2 = simplex[1]
    v3 = simplex[2]
    dist1 = enorm(a(v1[:-1]) - a(point[:2]))
    dist2 = enorm(a(v2[:-1]) - a(point[:2]))
    dist3 = enorm(a(v3[:-1]) - a(point[:2]))
    obj1 = max(v1[-1]['obj'][0] - L[0]*dist1,
               v2[-1]['obj'][0] - L[0]*dist2,
               v3[-1]['obj'][0] - L[0]*dist3)
    obj2 = max(v1[-1]['obj'][1] - L[1]*dist1,
               v2[-1]['obj'][1] - L[1]*dist2,
               v3[-1]['obj'][1] - L[1]*dist3)
    return [obj1, obj2]


def line_and_cone_intersection(cone_vertex, line_points, L):
    '''Based on 2014-10-20 PhDReport (9)-(19) equations.'''
    B = cone_vertex
    A, C = line_points
    a1, a2, a3 = A
    b1, b2, b3 = B
    c1, c2, c3 = C
    delta1 = b1 - a1
    delta2 = b2 - a2
    delta3 = b3 - a3
    n1 = c1 - a1
    n2 = c2 - a2
    n3 = c3 - a3
    d = n3**2/L**2 - n1**2 - n2**2
    e = 2*(n3*delta3/L**2 - n1*delta1 - n2*delta2)
    f = delta3**2/L**2 - delta1**2 - delta2**2
    det = e**2 - 4*d*f
    if det < 0:
        return []
    if det == 0:
        mu = -e / (2*d)
        return [[n1*mu + a1, n2*mu + a2, n3*mu + a3]]
    if round(d, 8) == 0:        # If d is zero, then its not quadratic equation
        mu = f / float(e)       # We solve simple linear equation
        return [[n1*mu + a1, n2*mu + a2, n3*mu + a3]]
    mu1 = -e + sqrt(det) / (2*d)
    mu2 = -e - sqrt(det) / (2*d)
    s1 = [n1*mu1 + a1, n2*mu1 + a2, n3*mu1 + a3]
    s2 = [n1*mu2 + a1, n2*mu2 + a2, n3*mu2 + a3]
    return [s1, s2]

def find_mins_ABC(simplex, L):
    '''Using Nelder-Mead optimization method find minimum of 3d Lipschitz lower bound.'''
    mins_ABC = []

    def is_in_region(t, p):
        '''Checks if point is in the triangle region using Barycentric coordinates:
        www.farinhansford.com/dianne/teaching/cse470/materials/BarycentricCoords.pdf'''
        A = det(a([[t[0][0], t[1][0], t[2][0]], [t[0][1], t[1][1], t[2][1]], [1., 1., 1.]]))
        A1 = det(a([[p[0], t[1][0], t[2][0]], [p[1], t[1][1], t[2][1]], [1, 1, 1]]))
        A2 = det(a([[t[0][0], p[0], t[2][0]], [t[0][1], p[1], t[2][1]], [1, 1, 1]]))
        A3 = det(a([[t[0][0], t[1][0], p[0]], [t[0][1], t[1][1], p[1]], [1, 1, 1]]))
        # Warnning: division by zero encountered
        u = A1 / A
        v = A2 / A
        w = A3 / A
        if u >= 0 and v >= 0 and w >= 0:
            return True
        return False

    def get_simplex_lower_bound_minimum(X):     # How to set current simplex for one argument function?
        l = 1.
        if not is_in_region(s, X):
            return float('inf')
        return max([s[0][-1] - l*enorm(a(s[0][:-1]) - a(X)),
                    s[1][-1] - l*enorm(a(s[1][:-1]) - a(X)),
                    s[2][-1] - l*enorm(a(s[2][:-1]) - a(X))])

    for i in xrange(len(simplex[0][-1]['obj'])):   # Iterates through objective indexes
        x = a([v[:-1] for v in simplex])
        y = a([v[-1]['obj'][i] for v in simplex])
        s = np.column_stack((x,y))

        res = minimize(get_simplex_lower_bound_minimum,
                       s.mean(0)[:-1],
                       method='nelder-mead',
                       options={'xtol': 1e-8, 'disp': False})

        mins_ABC.append(list(res.x) + [res.fun])
    return mins_ABC

#########   Tolerance types   #########
def get_max_vertex_obj_dist_to_obj_lb_min_points(simplex, L):
    '''Maksimalus viršūnės tikslo euklidinis atstumas iki min lb.obj1 ir min lb.obj2 tikslų.
    Tolerancija ne mažėjanti.
    '''
    v1 = simplex[0]
    v2 = simplex[1]
    v3 = simplex[2]

    mins_ABC = find_mins_ABC([v1,v2,v3], L)

    # Find other objective value at this mins_ABC points
    p1_objs = get_lower_bound_values_3D(simplex, mins_ABC[0])
    p2_objs = get_lower_bound_values_3D(simplex, mins_ABC[1])

    return max([min([enorm(a(v1[-1]['obj']) - a(p1_objs)),
                     enorm(a(v1[-1]['obj']) - a(p2_objs))]),
                min([enorm(a(v2[-1]['obj']) - a(p1_objs)),
                     enorm(a(v2[-1]['obj']) - a(p2_objs))]),
                min([enorm(a(v3[-1]['obj']) - a(p1_objs)),
                     enorm(a(v3[-1]['obj']) - a(p2_objs))])])

def get_max_vertex_obj_dist_to_obj_lb_mins(simplex, L):
    '''Maksimalus viršūnės tikslo euklidinis atstumas iki (min lb.obj1, min lb.obj2).'''
    v1 = simplex[0]
    v2 = simplex[1]
    v3 = simplex[2]

    mins_ABC = find_mins_ABC([v1,v2,v3], L)
    diss = [mins_ABC[0][-1], mins_ABC[1][-1]]
    return max([enorm(a(v1[-1]['obj']) - a(diss)),
                enorm(a(v2[-1]['obj']) - a(diss)),
                enorm(a(v3[-1]['obj']) - a(diss))])

def get_max_vertex_obj_city_block_dist_to_obj_lb_mins(simplex, L):
    '''Maksimalus viršūnės tikslo miesto kvartalų atstumas iki (min lb.obj1, min lb.obj2).'''
    v1 = simplex[0]
    v2 = simplex[1]
    v3 = simplex[2]

    mins_ABC = find_mins_ABC([v1,v2,v3], L)
    # diss = [mins_ABC[0][-1], mins_ABC[1][-1]]
    diss = [min([mins_ABC[0][-1], v1[-1]['obj'][0], v2[-1]['obj'][0], v3[-1]['obj'][0]]),
            min([mins_ABC[1][-1], v1[-1]['obj'][1], v2[-1]['obj'][1], v3[-1]['obj'][1]])]
    return max([city_block_norm(a(v1[-1]['obj']) - a(diss)),
                city_block_norm(a(v2[-1]['obj']) - a(diss)),
                city_block_norm(a(v3[-1]['obj']) - a(diss))])

def get_approx_lb_min(simplex, L):
    def get_approx_min_nelder_mead(simplex, L, verts=3):
        '''Using Nelder-Mead optimization method find minimum of 3d Lipschitz lower bound.'''

        def is_in_simplex(simplex, point):
            '''nD->1D Based on Barycentric coordinates: https://en.wikipedia.org/wiki/Barycentric_coordinate_system.
            http://math.stackexchange.com/questions/1226707/how-to-check-if-point-x-in-mathbbrn-is-in-a-n-simplex/1226825#1226825
            '''
            rl = a(simplex[-1])
            bar_coefs = inv((simplex[:-1] - rl).T).dot(a(point) - rl)

            bar_coefs = hstack((bar_coefs, 1. - sum(bar_coefs)))

            # restored_point = None
            # for i, coef in enumerate(bar_coefs):
            #     if restored_point is None:
            #         restored_point = coef * (a(simplex[i]))
            #     else:
            #         restored_point += coef * (a(simplex[i]))
            # for i in range(len(point)):
            #     if round(point[i], 8) != round(restored_point[i], 8):
            #         raise ValueError('Barycentric coordinates determined incorrectly during is_in_simplex check.', point, restored_point)

            if sum(bar_coefs) <= 1 and min(bar_coefs) >= 0:
                return True
            return False

        def get_simplex_lower_bound_minimum(X, simp, L, verts=3):     # How to set current simplex for one argument function?
            '''Uses 3 vertex information'''
            if not is_in_simplex(simp[:,:-1], X):
                return float('inf')

            lb_values = []
            for i, v in enumerate(simp):
                if i < verts:
                    lb_values.append(v[-1] - L*enorm(a(v[:-1]) - a(X)))
            return max(lb_values)

        x = a([v[:-1] for v in simplex])
        y = a([v[-1]['obj'][0] for v in simplex])
        s = np.column_stack((x, y))

        res = minimize(get_simplex_lower_bound_minimum,
                       x.mean(0),
                       method='nelder-mead',
                       options={'xtol': 1e-8, 'disp': False},
                       args=(s, L, verts))
        return res.fun  # res.x

    def get_approx_min_longest_edge(simplex, L):
        '''n->1 l.b.m.approx.: max(f(A) - L*|AB|, f(B) - L*|AB|)'''
        A = simplex[0]
        B = simplex[1]

        AB_dist = l2norm(A[:-1], B[:-1])
        return max([A[-1]['obj'] - L*AB_dist,  B[-1]['obj'] - L*AB_dist])

    def get_approx_min_max_angle(simplex, L):
        '''nD->nD Approximates nD simplexes lower bound minimum by extending longest age by 1/cos(max angle)'''
        def get_angle(A, B, C):
            '''Finds angle in radians between AB and BC vectors'''
            vec1 = a(A) - a(B)
            vec2 = a(C) - a(B)
            return np.arccos(np.dot((vec1), (vec2))/ (enorm(vec1) * enorm(vec2)))     # radians to degree: * 180/np.pi

        # Choose longest edge vertexes
        A = simplex[0]
        B = simplex[1]

        # Find maximum angles for each vertex
        A_angles = []
        B_angles = []

        for V in nm(simplex):
            if V != A and V != B:
                A_angles.append(get_angle(B[:-1], A[:-1], V[:-1]))
                B_angles.append(get_angle(A[:-1], B[:-1], V[:-1]))

        max_A_angle = max(A_angles)
        max_B_angle = max(B_angles)

        v1 = simplex[0]
        v2 = simplex[1]

        if type(simplex[-1]) == dict and simplex[-1].has_key['mins_AB']:
            mins_AB = simplex[-1]['mins_AB']
        else:
            mins_AB = find_mins_AB(simplex, L)

        return min([
            v1[-1]['obj'][0] - L*l2norm(nm(v1), mins_AB[0][:-1]) / np.cos(max_A_angle),
            v2[-1]['obj'][0] - L*l2norm(nm(v2), mins_AB[0][:-1]) / np.cos(max_B_angle)
        ])

    if type(simplex[-1]) == dict and simplex[-1].has_key('approx_min_ABC'):
        return simplex[-1]['approx_min_ABC']

    # return get_approx_min_longest_edge(simplex, L)
    return get_approx_min_max_angle(simplex, L)
    # return get_approx_min_nelder_mead(simplex, L, verts=2)


def get_tolerance(simplex, L):
    if type(simplex[-1]) == dict and simplex[-1].has_key('approx_min_ABC'):
        lbm = simplex[-1].get('approx_min_ABC')
    else:
        lbm = get_approx_lb_min(simplex, L)

    min_dist = None
    for v in simplex[:-1]:
        obj_dist = l2norm(v[-1]['obj'], lbm)
        if min_dist is None or obj_dist < min_dist:
            min_dist = obj_dist
    return min_dist


def convex_hull(x, y, it=0):
    '''2D convex hull algorithm.'''
    def nextv(v, m):
        if v == m:
            return 0
        return v + 1
    def predv(v, m):
        if v == 0:
            return m
        return v - 1
    x = x[:]
    y = y[:]
    m = len(x) - 1
    if len(x) != len(y):
        raise ValueError('Convex hull input dimensions must match')
    if m == 1:
        return [0, 1]
    if m == 0:
        return [0]
    # if len(x) == 0:
    #     return []
    START = 0
    v = START
    w = len(x) - 1
    flag = 0
    leftturn = None
    h = range(len(x))
    while (nextv(v, m) != START) or (flag == 0):
        if nextv(v, m) == w:
            flag = 1
        a = v
        b = nextv(v, m)
        c = nextv(nextv(v, m), m)
        det_val = det([[x[a], y[a], 1], [x[b], y[b], 1], [x[c], y[c], 1]])
        if det_val >= 0:
            leftturn = 1
        else:
            leftturn = 0
        if leftturn:
            v = nextv(v, m)
        else:
            j = nextv(v, m)
            x.pop(j)
            y.pop(j)
            h.pop(j)
            m -= 1
            w -= 1
            v = predv(v, m)
    return h


def select_simplexes_to_divide(simplexes, it=0, mirror_division=False):
    sorted_simplexes = sorted(simplexes, key=lambda x: (x[-1]['value'], x[-1]['size']))
    d = [s[-1]['size'] for s in sorted_simplexes]
    f = [s[-1]['value'] for s in sorted_simplexes]
    f_min = min(f)

    # Find best simplex:  (F - f_min + E)./D   # E = max(epsilon*abs(f_min),1E-8))
    # Kiekvienam simpleksui apskaičiuoti dydį: (F - f_min + E)./D
    met = [(s[-1]['value'] - f_min + 10**-8) / s[-1]['size'] for s in sorted_simplexes]
    min_met = min(met)
    i_min = met.index(min_met)
    # i_min = f.index(f_min)

    # For each division find min function value.
    ## globalūs indeksai reikalingi elementų su minimalia reikšme
    unique_divisions = sorted(list(set([s[-1]['size'] for s in simplexes])))
    ud_min_ids = []
    ud_i_min_id = None
    for division in unique_divisions:
        simplexes_with_division = [s for s in simplexes if s[-1]['size'] == division]
        min_obj = min(simplexes_with_division, key=lambda o: o[-1]['value'])
        ud_i_min_id = len(ud_min_ids)
        j = sorted_simplexes.index(min_obj)
        if min_obj == j:
            ud_i_min_id = len(ud_min_ids)
        ud_min_ids.append(j)

    if len(ud_min_ids) > 2 and i_min != ud_min_ids[-1]:
    # if len(ud_min_ids) - ud_i_min_id > 1:
        # This comparison solves problem when the largest simplex has lowest value i_min == ud_min_ids[-1]
        # Find points below given line.
        S = []
        a1 = d[i_min]
        b1 = f[i_min]
        a2 = d[ud_min_ids[-1]]
        b2 = f[ud_min_ids[-1]]

        slope = (b2 - float(b1))/(a2 - a1)
        # if it == 4:
        #     import ipdb; ipdb.set_trace()
        const = b1 - slope*a1
        for i in range(len(ud_min_ids)):
            j = ud_min_ids[i]
            if f[j] <= slope*d[j] + const + 10**-12:
                S.append(j)

        dd = [d[i] for i in S]
        ff = [f[i] for i in S]
        # Find convex hull for them.
        h = convex_hull(dd, ff, it)
        ids = [S[i] for i in h]
    else:
        ids = ud_min_ids

    # Atmesk visus elementus, kurie neatitinka sąlygos:
    # f - slope*d > f_min - epsilon*abs(f_min)
    remove = []
    for i in range(len(ids)-1):
        i_a = ids[len(ids) - i - 1]
        i_b = ids[len(ids) - i - 2]
        a1 = d[i_a]
        b1 = f[i_a]
        a2 = d[i_b]
        b2 = f[i_b]
        slope = (b2 - float(b1))/(a2 - a1)
        const = b1 - slope*a1
        if const > f_min - 0.0001*abs(f_min):
            remove.append(i_b)
        # if len(remove):
        #     ax1, ax2 = show_potential(simplexes, show=False)
        #     ax2.plot([0, a1], [const, slope*a1+const], 'g-')
        #     ax2.plot([0], [f_min - 0.0001*abs(f_min)], 'ro')
        #     ax2.plot([0], [const], 'go')
        #     plt.show()
    for i in remove:
        ids.remove(i)


    # Reikia išrinkti visus elementus turinčius nurodytą reikšmę
    selected = []
    for i in ids:
        for s in sorted_simplexes:
            if (s[-1]['value'] == f[i] and s[-1]['size'] == d[i]
                                         and s not in selected):
                selected.append(s)

    if mirror_division:      # Somehow does not have any effect, needs more inspection
        for s in selected:
            simplexes_with_same_longest_edge = get_simplexes_with_given_longest_edge(simplexes, s[0], s[1])
            for swle in simplexes_with_same_longest_edge:
                if swle not in selected and s[-1]['size'] == swle[-1]['size']:
                    selected.append(swle)

    # from pprint  import pprint
    # print 'Simplexes'
    # pprint(simplexes)
    # print 'Worst simplex', max(simplexes, key=lambda x: x[-1]['value'])
    # print 'Worst neibours'
    # pprint([s for s in simplexes if [0.75, 0.75] in [v[:-1] for v in s[:-1]]])
    # # print 'Best simplexes'
    # # pprint([s for s in simplexes if [1., 0.25] in [v[:-1] for v in s[:-1]]])
    # # print '-----'
    # # pprint([s for s in simplexes if [1., 0.125] in [v[:-1] for v in s[:-1]]])
    # print 'Simplex best value', f
    # print 'Simplex diameter', d
    # show_potential(simplexes, selected)
    return selected
    # Find all simplexes with higher then minimum size:
    # with D(i) >= D(i_min) and F(i) is the minimum function value for the current distance



def get_division_point(A, B, f):
    return add_objective_values(list((a(nm(A)) + a(nm(B)))/2.), f)

def dominates(p, q):
    '''Point p dominates q if all its objectives are better or equal'''
    dominates = False
    for i in xrange(len(p[-1]['obj'])):
        if p[-1]['obj'][i] > q[-1]['obj'][i]:
            return False
        elif p[-1]['obj'][i] < q[-1]['obj'][i]:
            dominates = True
    return dominates

def get_pareto_front(X):
    '''Returns non dominated points.'''
    P = []
    for x in X:
        dominated = False
        for xd in X:
            if x != xd:
                if dominates(xd, x):
                    dominated = True
        if not dominated and x not in P:
            P.append(x)
    return P


def update_pareto_front(pareto_front, division_point):
    pareto_front.append(division_point)
    return get_pareto_front(pareto_front)

def remove_dominated_simplexes(simplexes):
    min_vertex_obj = None
    for simplex in simplexes:
        for vertex in simplex[:-1]:
            vertex_obj = vertex[-1]['obj'][0]
            if min_vertex_obj is None or vertex_obj < min_vertex_obj:
                min_vertex_obj = vertex_obj

    to_remove = []
    for i, approx_min_ABC in enumerate([s[-1]['approx_min_ABC'] for s in simplexes]):
        if approx_min_ABC > min_vertex_obj:
            to_remove.append(i)
    return [simplex for i, simplex in enumerate(simplexes) if i not in to_remove]
    # min_vertex_obj = min(simplexes, key=lambda x: x[-1]['approx_min_ABC'])


def get_simplexes_with_given_longest_edge(simplexes, A, B):
    with_the_edge = []
    for s in simplexes:
        if A in s and B in s:
            with_the_edge.append(s)
    return with_the_edge


def should_stop(actual_f_min, found_min, error):
    if actual_f_min == 0:
        return (found_min*100 < error)
    return (found_min - actual_f_min)/abs(actual_f_min)*100 < error

def disimpl_2v(f, lb, ub, error, max_f_calls, f_min, mirror_division=False):
    simplexes = triangulate([0.]*len(lb), [1.]*len(ub))
    # simplexes = [ # Disimpl_v Branin example
    #     [[0., 1.], [0.25, 0.75], [0.5, 1.]],
    #     [[0., 1.], [0.25, 0.75], [0., 0.5]],
    #     [[0., 0.5], [0.25, 0.75], [0.5, 0.5]],
    #     [[0.25, 0.75], [0.5, 1.], [0.5, 0.5]],
    #     [[0.5, 1.], [0.5, 0.5], [0.75, 0.75]],
    #     [[0.5, 1.], [0.75, 0.75], [1., 1.]],
    #     [[0.75, 0.75], [1., 1.], [1., 0.5]],
    #     [[0.5, 0.5], [0.75, 0.75], [1., 0.5]],
    #     [[0., 0.], [0., 0.5], [0.25, 0.25]],
    #     [[0., 0.5], [0.25, 0.25], [0.5, 0.5]],
    #     [[0., 0.], [0.25, 0.25], [0.25, 0.]],
    #     [[0.25, 0.25], [0.25, 0.], [0.5, 0.]],
    #     [[0.5, 0.5], [0.25, 0.25], [0.5, 0.25]],
    #     [[0.25, 0.25], [0.5, 0.25], [0.5, 0.]],
    #     [[0.5, 0.5], [0.75, 0.25], [1., 0.5]],
    #     [[0.5, 0.], [0.75, 0.25], [0.75, 0.]],
    #     [[0.75, 0.], [0.75, 0.25], [1., 0.]],
    #     [[0.5, 0.25], [0.5, 0.5], [0.75, 0.25]],
    #     [[0.5, 0.], [0.5, 0.25], [0.75, 0.25]],
    #     [[0.75, 0.25], [0.875, 0.25], [0.875, 0.125]],
    #     [[0.75, 0.25], [0.875, 0.25], [0.875, 0.375]],
    #     [[0.875, 0.25], [0.875, 0.375], [1., 0.25]],
    #     [[0.875, 0.375], [1., 0.25], [1., 0.375]],
    #     [[0.875, 0.375], [1., 0.375], [1., 0.5]],
    #     [[0.875, 0.25], [1., 0.25], [0.875, 0.125]],
    #     [[0.875, 0.125], [1., 0.125], [1., 0.25]],
    #     [[0.875, 0.125], [1., 0.125], [1., 0.]],
    # ]
    points = find_objective_values_for_vertexes(simplexes, f)
    simplexes = find_meta_for_simplexes(simplexes)

    pareto_front = get_pareto_front(points)

    i = 0
    done = False
    while True: # i < max_iters:
        if i == 0:
            simplexes_to_divide = copy(simplexes)
        else:
            simplexes_to_divide = select_simplexes_to_divide(simplexes, i, mirror_division)
        # show_potential(simplexes, simplexes_to_divide)

        # simplex_to_divide[-1]['hash'] = hash(str(simplex_to_divide[:-1]))

        # Chose division point
        # print i, 'will divide', len(simplexes_to_divide)
        # show_partitioning(simplexes)
        # show_potential(simplexes, simplexes_to_divide)

        for simplex_to_divide in simplexes_to_divide:
            division_point = get_division_point(*(simplex_to_divide[:2] + [f]))

            # Update pareto front
            # update_pareto_front(pareto_front, division_point)

            ### When dividing longest edge - divide and mirror simplexes
            # Get simplexes, which have the same longes edge

            # Add new simplexes, remove divided one
            new_simplex1 = sort_vertexes_longest_edge_first([simplex_to_divide[0], division_point] + simplex_to_divide[2:-1])
            new_simplex2 = sort_vertexes_longest_edge_first([simplex_to_divide[1], division_point] + simplex_to_divide[2:-1])
            new_simplex1.append({
                'value':  find_simplex_value(new_simplex1),
                'size': find_simplex_size(new_simplex1),
                                 # 'mins_AB': find_mins_AB(new_simplex1, L),
                                 # 'approx_min_ABC': get_approx_lb_min(new_simplex1, L),
                                 # 'parent_hash': hash(str(simplex_to_divide[:-1])),
                                 # 'hash': hash(str(new_simplex1[:-1])),
                                 # 'tolerance': get_tolerance(new_simplex2, L),
                                 })
            new_simplex2.append({
                'value':  find_simplex_value(new_simplex2),
                'size': find_simplex_size(new_simplex2),
                                 # 'mins_AB': find_mins_AB(new_simplex2, L),
                                 # 'approx_min_ABC': get_approx_lb_min(new_simplex2, L),
                                 # 'parent_hash': hash(str(simplex_to_divide[:-1])),
                                 # 'hash': hash(str(new_simplex2[:-1])),
                                 # 'tolerance': get_tolerance(new_simplex2, L),
                                 })

            index = simplexes.index(simplex_to_divide)
            simplexes.remove(simplex_to_divide)
            simplexes.insert(index, new_simplex1)
            simplexes.insert(index, new_simplex2)

            found_min = min([min([v[-1]['obj'][0] for v in s[:-1]]) for s in simplexes])
            # if should_stop(f_min, found_min, error):
            #     done = True
            #     break

        if should_stop(f_min, found_min, error):
            done = True
            break

        if f.calls >= max_f_calls:
            done = True
            break

            # remove_dominated_simplexes(simplexes)
        i += 1
        print f.calls
        if done:
            break
        # print 'Number of simplexes', len(simplexes), 'Function calls:', f.calls
        # print i, tolerance
        # print '%d.  simplex approx.l.b.min.: %.14f' % (i, simplex_to_divide[-1]['approx_min_ABC'])
        # print '%d.  tolerance:  %.14f' % (i,  min([s[-1]['tolerance'] for s in simplexes]))
    return pareto_front, simplexes, f


if __name__ == '__main__':
    from experiments import functions, get_D, get_lb, get_ub, get_min, gkls_function

    max_f_calls = 10000
    error = 0.01
    mirror_division = False

    for gkls_cls in range(1, 9):
        for gkls_fid in range(1, 101):
            f_name = 'gkls_cls%d_%d' % (gkls_cls, gkls_fid)
            D = get_D(f_name)
            lb = get_lb(f_name, D)
            ub = get_ub(f_name, D)
            f = wrap(gkls_function, lb, ub, gkls_cls=gkls_cls, gkls_fid=gkls_fid)   # f = wrap(dict(functions)[f_name], lb, ub)
            print f_name + ':'
            min_x = get_min(f_name, D)[:-1]
            min_f = get_min(f_name, D)[-1]
            # draw_3d_objective_function(branin, lb, ub)
            start = datetime.now()
            pareto_front, simplexes, f = disimpl_2v(f, lb, ub, error, max_f_calls, min_f, mirror_division)
            end = datetime.now()
            print '   ', f.calls, f.min_f, f.min_x, f_name
            print "Calls:", f.calls, 'Found min f:', f.min_f, 'at x:', f.min_x[:-1]
            print "f*:", min_f, "x*:", min_x
            print "Duration", end-start
            exit()
        # show_lower_pareto_bound(simplexes)
        # show_partitioning(simplexes)
        # show_potential(simplexes)
        # show_pareto_front(pareto_front)
