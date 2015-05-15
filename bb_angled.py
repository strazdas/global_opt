#! coding: utf-8
'''
Dviejų kintamųjų, dviejų tikslo funkcijų problemos sprendimo algoritmas, kurio
strategija dalinti stačiais trikmapiais, ilgiausią kraštinę dalinant pusiau.

Išbandytos trys tolerancijos apskaičiavimo strategijos:
    1. maksimalus euklidinis viršūnių atstumas iki minimumų (netinkamas)
    2. maksimalus euklidinis viršūnių atstumas iki minimalių reikšmių (tinkamas)
    3. maksimalus citų block viršūnių atstumas iki minimalių reikšmių (tinkamas)

Pradinė erdvė padalinama į keturis stačiuosius trikampius.
Toliau dalinami trikmapiai, kurių apatinės ribos minimumas mažiausias.
'''
from copy import copy
from itertools import permutations
from numpy import array as a, sqrt, isnan, isinf
from utils import enorm, city_block_norm, show_pareto_front, show_partitioning,\
                  draw_bounds, remove_dominated_subinterval, show_lower_pareto_bound,\
                  draw_simplex_3d_euclidean_bounds, draw_two_objectives_for_2d_simplex,\
                  show_simplex_minimum_dissmatch, is_in_region
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from numpy.linalg import det


def f1(X):
    '''First problem form multiobjective-bivariate algorithm paper 5 chapter'''
    return X[0]

def f2(X):
    return (min(abs(X[0] - 1), 1.5 - X[0]) + X[1] + 1) / sqrt(2)

def triangulate(lb, ub):
    '''Triangulates the given hyperrectangle using combinatoric triangulation.'''
    # center_point = [(lb[0]+ub[0])/2., (lb[1]+ub[1])/2.]
    # triangles = [
    #         [lb,[lb[0], ub[0]], center_point],
    #         [lb,[ub[1], lb[1]], center_point],
    #         [[lb[0], ub[0]],ub, center_point],
    #         [[ub[1], lb[1]],ub, center_point]]

    triangles = []
    for t in permutations(xrange(D)):
        vertexes = [copy(lb)]
        triangles.append(vertexes)
        for i in xrange(D):
            vertexes.append([])
            for j in xrange(D):
                vertexes[-1].append(vertexes[-2][j])
            vertexes[-1][t[i]] = ub[t[i]]
    return triangles

def find_objective_values_for_vertexes(simplexes):
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
    return [add_objective_values(p) for p in points]

def add_objective_values(x):
    '''Adds objective values to meta information for a single point (e.g. simplex vertex).'''
    if type(x) == list:
        x.append({'obj': (f1(x), f2(x))})
        return x
    return [x, {'obj': (f1(x), f2(x))}]

def find_mins_AB_and_tolerance_for_simplexes(simplexes):
    '''Groups points into intervals and computes tolerance for each interval.'''
    for simplex in simplexes:
        sort_vertexes_longest_edge_first(simplex)
        if type(simplex[-1]) == dict:
            simplex[-1]['mins_AB'] = find_mins_AB(simplex, L)
            simplex[-1]['mins_ABC'] = find_mins_ABC(simplex, L)
            simplex[-1]['tolerance'] = get_tolerance(simplex, L)
            simplex[-1]['approx_mins_ABC'] = get_approx_min_ABC(simplex)
        else:
            simplex.append({'mins_AB': find_mins_AB(simplex, L),
                            'mins_ABC': find_mins_ABC(simplex, L),
                            'tolerance': get_tolerance(simplex, L),
                            'approx_mins_ABC': get_approx_min_ABC(simplex),
                            })
    return simplexes

def sort_vertexes_longest_edge_first(simplex):
    '''Motves longest edge vertexes to the simplex vertex list beggining.'''
    edge_lengths = []
    for i in xrange(len(simplex)):
        edge_lengths.append(enorm(a(simplex[i-1][:-1]) - a(simplex[i][:-1])))
    max_len_index = edge_lengths.index(max(edge_lengths))
    top_vertex = simplex.pop(max_len_index - 2)
    simplex.append(top_vertex)
    return simplex

def find_mins_AB(simplex, L):
    '''
    Finds AB' and B'A intersection, where A, B are 
    t - triangle (simplex).
    y - objective values for each vertex.
    Returns lower Lipschitz bound minimum for the first edge (made from first
    and second vetexes).
    '''
    mins_AB = []
    t = a([v[:-1] for v in simplex])
    y = a([v[-1]['obj'] for v in simplex])

    # For each objective find the minimum AB point
    for i in xrange(y.shape[1]):
        f = y[:,i]    # function values
        l = L[i]      # L value
        A = a([t[0][0], t[0][1], f[0]])  # Point A
        B = a([t[1][0], t[1][1], f[1]])  # Point B

        D = a([B[0], B[1], A[2]-l*enorm(a([B[0]-A[0], B[1]-A[1]]))])  # Point B'
        E = a([A[0], A[1], B[2]-l*enorm(a([A[0]-B[0], A[1]-B[1]]))])  # Point A'
        n1 = a(D) - a(A)  # l1 krypties vektorius
        n2 = a(E) - a(B)  # l2 krypties vektorius
        p1 = a(A)
        p2 = a(B)

    ## Tinka ne visos koordinatės, nes kai kurių koordinačių atžvilgių tiesės sutampa.
        s = (p2[2]*n1[0] - p1[2]*n1[0] - p2[0]*n1[2] + p1[0]*n1[2]) / (n2[0]*n1[2] - n2[2]*n1[0])
        if np.isnan(s):
            s = (p2[2]*n1[1] - p1[2]*n1[1] - p2[1]*n1[2] + p1[1]*n1[2]) / (n2[1]*n1[2] - n2[2]*n1[1])
        if np.isnan(s):
            s = (p2[1]*n1[0] - p1[1]*n1[0] - p2[0]*n1[1] + p1[0]*n1[1]) / (n2[0]*n1[1] - n2[1]*n1[0])
        X = n2 * s + p2
        mins_AB.append(X)
    return mins_AB


## Tolerance methods
def get_lower_bound_values_3D(simplex, point):
    '''Returns bouth objective values at a given point in simplex'''
    L = [1., 1.]
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
        u = A1 / A
        v = A2 / A
        w = A3 / A
        if u >= 0 and v >= 0 and w >= 0:
            return True
        return False

    def get_simplex_lower_bound_minimum(X, nr_of_verts=2):     # How to set current simplex for one argument function?
        l = 1.
        if not is_in_region(s, X):
            return float('inf')
        if nr_of_verts == 2:
            return max([s[0][-1] - l*enorm(a(s[0][:-1]) - a(X)),
                        s[1][-1] - l*enorm(a(s[1][:-1]) - a(X))])
        elif nr_of_verts == 3:
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





## def find_mins_ABC(simplex, L, j=0):
##     '''Warning: this method is theoretically wrong. Do not use this method.'''
##     '''For each objective finds minimum simplex lower Lipshitz bound values.
##     simplex - triangle vertexes list. Each vertex has objective values attached.
##     L - Lipschitz constant.
##     Returns minimum point for each objective function.
##     '''
##     mins_ABC = []
## 
##     for i in xrange(len(simplex[0][-1]['obj'])):
##         ## extract objective values
##         y = a([v[-1]['obj'][i] for v in simplex])
##         l = L[i]
##         x = a([v[:-1] for v in simplex])
## 
##         a1, b1 = x[0]
##         a2, b2 = x[1]
##         a3, b3 = x[2]
##         c1, c2, c3 = y
## 
##         # Find point of lower bound cones intersection as shown in Gražina paper.
##         l32 = a3**2 - a2**2 + b3**2 - b2**2 + c2**2 - c3**2
##         l21 = a2**2 - a1**2 + b2**2 - b1**2 + c1**2 - c2**2
##         k = (a3 - a2)/(a2 - a1)
## 
##         my = (l32 - k*l21)/(2*(b3 - b2 + b1*k - b2*k))
##         mx = l21 - 2*my*(b2-b1)/(2*(a2 - a1))
##         import ipdb; ipdb.set_trace()
##         pass
##         # mobj = get_lower_bound_value([mx, my], L)
##         ## mins_ABC.append([mx, my, mobj])
## 
##         ## intersects = []
##         ## cone_vertex = list(x[1]) + [y[1]]
##         ## line_points = [list(x[0]) + [y[0]], list(x[2]) + [y[0] - l*enorm(x[0]-x[2])]]       # Use only two points check in yearly report
##         ## intersects += line_and_cone_intersection(cone_vertex, line_points, l)
##         ## cone_vertex = list(x[0]) + [y[0]]
##         ## line_points = [list(x[1]) + [y[1]], list(x[2]) + [y[1] - l*enorm(x[1]-x[2])]]       # Use only two points check in yearly report
##         ## intersects += line_and_cone_intersection(cone_vertex, line_points, l)
##         ## # Add C point to intersects, before finding mins_ABC
##         ## intersects += [list(x[2]) + [y[2]]]
##         ## ## Remove solutions with NaN and Inf objective values
##         ## intersects = [s for s in intersects if not isnan(s[-1]) and not isinf(s[-1])]
##         ## ## Find point with lowest objective value
##         ## mins_ABC.append(min(intersects, key=lambda x: x[-1]))
##     return mins_ABC

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



def get_simplex_analytical_lb_min(simplex, L=[1., 1.]):
    '''Warning: this method causes increasing tolerance.
    Warning: This method does not work, returns invalid solutions sometimes.
    '''
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

    def is_in_region(p, t):
        '''Checks if point is in the triangle region using Barycentric coordinates:
        www.farinhansford.com/dianne/teaching/cse470/materials/BarycentricCoords.pdf'''
        A = det(a([[t[0][0], t[1][0], t[2][0]], [t[0][1], t[1][1], t[2][1]], [1., 1., 1.]]))
        A1 = det(a([[p[0], t[1][0], t[2][0]], [p[1], t[1][1], t[2][1]], [1, 1, 1]]))
        A2 = det(a([[t[0][0], p[0], t[2][0]], [t[0][1], p[1], t[2][1]], [1, 1, 1]]))
        A3 = det(a([[t[0][0], t[1][0], p[0]], [t[0][1], t[1][1], p[1]], [1, 1, 1]]))
        u = A1 / A
        v = A2 / A
        w = A3 / A
        if u >= 0 and v >= 0 and w >= 0:
            return True
        return False

    def find_mins_ABC(simplex, L):
        '''For each objective finds minimum simplex lower Lipshitz bound values.
        simplex - triangle vertexes list. Each vertex has objective values attached.
        L - Lipschitz constant.
        Returns minimum point for each objective function.
        '''
        mins_ABC = []

        for i in xrange(len(simplex[0][-1]['obj'])):
            ## extract objective values
            y = a([v[-1]['obj'][i] for v in simplex])
            l = L[i]
            x = a([v[:-1] for v in simplex])

            intersects = []
            cone_vertex = list(x[1]) + [y[1]]
            line_points = [list(x[0]) + [y[0]], list(x[2]) + [y[0] - l*enorm(x[0]-x[2])]]       # Use only two points check in yearly report
            intersects += line_and_cone_intersection(cone_vertex, line_points, l)
            cone_vertex = list(x[0]) + [y[0]]
            line_points = [list(x[1]) + [y[1]], list(x[2]) + [y[1] - l*enorm(x[1]-x[2])]]       # Use only two points check in yearly report
            intersects += line_and_cone_intersection(cone_vertex, line_points, l)
            # Add C point to intersects, before finding min_ABC
            intersects += [list(x[2]) + [y[2]]]
            ## Remove solutions with NaN and Inf objective values
            intersects = [s for s in intersects if not isnan(s[-1]) and not isinf(s[-1])]
            ## Find point with lowest objective value
            min_ABC = None

            # if [2.875, 2.875, -2.6516504294495533] in intersects:
            #     import ipdb; ipdb.set_trace()
            for s in intersects:
                v1 = simplex[0]
                v2 = simplex[1]
                # np.sqrt((i1[0] - v1[0])**2 + (i1[1] - v1[1])**2) - v1[2] + i1[2]
                if (# is_in_region(s[:2], simplex) and  # <--- this check does not work
                    not round(np.sqrt((s[0] - v1[0])**2 + (s[1] - v1[1])**2) - v1[-1]['obj'][i] + s[2], 8) and
                    not round(np.sqrt((s[0] - v2[0])**2 + (s[1] - v2[1])**2) - v2[-1]['obj'][i] + s[2], 8)):
                    min_ABC = s
                    break
            mins_ABC.append(min_ABC)
        return mins_ABC

    return find_mins_ABC(simplex, L)  # [o[-1] for o in find_mins_ABC(simplex, L)]


def get_simplex_angle_approx_lb_min(simplex, L=[1., 1.]):
    '''Approximates ABC lower bound minimum points by decreasing AB lower bound by cosBAC.'''
    def get_alpha(A, B):
        return np.arccos(np.dot((A), (B))/ (enorm(A) * enorm(B)))     # radians to degree: * 180/np.pi

    v1 = simplex[0]
    v2 = simplex[1]
    A = a(simplex[0][:2])
    B = a(simplex[1][:2])
    C = a(simplex[2][:2])

    if type(simplex[-1]) == dict and simplex[-1].has_key['mins_AB']:
        mins_AB = simplex[-1]['mins_AB']
    else:
        mins_AB = find_mins_AB(simplex, L)

    mins_ABC = []
    for i in range(len(simplex[0][-1]['obj'])):
        min_ABC = min([   # Simpleksą aproksimuojantį minimumą turim
            v1[-1]['obj'][i] - L[i]*enorm(v1[:2] - a(mins_AB[i][:2]))/np.cos(get_alpha(C-A, B-A)),
            v2[-1]['obj'][i] - L[i]*enorm(v2[:2] - a(mins_AB[i][:2]))/np.cos(get_alpha(C-B, A-B))
        ])
        mins_ABC.append(min_ABC)
    return mins_ABC

def get_approx_min_ABC(simplex, L=[1., 1.]):
    return get_simplex_angle_approx_lb_min(simplex, L=[1., 1.])
    # return get_simplex_analytical_lb_min(simplex, L=[1., 1.])    # This method causes increasing tolerance.


def get_AB_cos_tolerance(simplex):
    '''Formulė šiai tolerancijai apskaičiuoti:
    dist_A = ||A[:2] - mins_AB[:2]|| / fiA
    dist_B = ||B[:2] - mins_AB[:2]|| / fiB

    min(y1 - dist_A*L, y2 - dist_B*L)
    '''
    if type(simplex[-1]) == dict and simplex[-1].has_key['approx_mins_ABC']:
        approx_mins_ABC = simplex[-1]['approx_mins_ABC']
    else:
        approx_mins_ABC = get_approx_min_ABC(simplex)
    if type(approx_mins_ABC[0]) == list:
        approx_mins_ABC = [o[-1] for o in approx_mins_ABC]
    return max([enorm(a(simplex[0][-1]['obj']) - a(approx_mins_ABC)),
                enorm(a(simplex[1][-1]['obj']) - a(approx_mins_ABC))])


def get_tolerance(simplex, L):
    # return get_max_vertex_obj_dist_to_obj_lb_min_points(simplex, L)
    # return get_max_vertex_obj_dist_to_obj_lb_mins(simplex, L)
    # return get_max_vertex_obj_city_block_dist_to_obj_lb_mins(simplex, L)
    return get_AB_cos_tolerance(simplex)

# def get_tolerance_2D_no_alpha(simplex, L):
#     '''Returns tolerance for 2D-simplex. Uses:
#     Longest edge distance.
#     Objective values at longest edge vertexes.
#     Lipschitz constant.
#     '''
#     L = [1., 1.]
#     #### OLD:
#     #### Find longest edge with L*alpha_hat tolerance.
#     ### mins_AB = find_mins_AB(simplex, list(a(L)*a(alphas)))
#     #### Patikrinti ar Lipšico konstanta tinkamai nustatyta.
#     #### Patikrinti ar kiekvieno apskaičiuojamo simplekso tolerancija apskaičiuoja teisingai.
# 
#     # Convert two variable points v1, v2 to single variable x1, x2
#     v1, v2 = simplex[0], simplex[1]
#     dist = enorm(a(v1[:-1]) - a(v2[:-1]))
#     x1 = [0.0, {'obj': v1[-1]['obj']}]
#     x2 = [dist, {'obj': v2[-1]['obj']}]
# 
#     # Remove dominated subintervals
#     x1, x2 = remove_dominated_subinterval(x1, x2)
#     y1 = x1[-1]['obj']
#     y2 = x2[-1]['obj']
# 
#     # Find crack points and tolerance
#     def lower_bound_cracks(x1=None, x2=None, y1=None, y2=None, dist=None):
#         '''Computes lower L bound crack points for the given interval.'''
#         if dist is None:
#             dist = enorm(x2-x1)
#         t1 = (y1[0]-y2[0])/(2.*L[0]) + dist/2.
#         t2 = (y1[1]-y2[1])/(2.*L[1]) + dist/2.
#         if t1 >= t2:
#             p1 = [y1[0] - L[0]*t2, y1[1] - L[1]*t2]
#             p2 = [y1[0] - L[0]*t1, y2[1] - dist*L[1] + L[1]*t1]
#         else:
#             p1 = [y1[0] - L[0]*t1, y1[1] - L[1]*t1]
#             p2 = [y2[0] - dist*L[0] + L[0]*t2, y1[1] - L[1]*t2]
#         return [p1, p2]
#     p1, p2 = lower_bound_cracks(y1=y1, y2=y2, dist=dist)
# 
#     # draw_bounds(x1, x2, L)
# 
#     return max([min((enorm(a(p1) - a(y1)), enorm(a(p2) - a(y1)))),
#                 min((enorm(a(p1) - a(y2)), enorm(a(p2) - a(y2))))])

def select_simplex_to_divide(simplexes):
    # What can be the division criteria?
    # I could really find the tolerance and divide by it.
    # STUB: method to map two values to one not used (should find one).

    # Should use tolerance.
    # return min(simplexes, key=lambda x: x[-1]['mins_AB'][0][-1])
    return max(simplexes, key=lambda x: x[-1]['tolerance'])

def get_division_point(A, B):
    return add_objective_values([(A[0] + B[0])/2., (A[1] + B[1])/2.])

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


def bb_angled_algorithm(lb, ub, L):
    # Padaliną leistinąją sritį į keturis stačiuosius trikampius (jeigu leistinoji sritis keturkampis)
    simplexes = triangulate(lb, ub)
    points = find_objective_values_for_vertexes(simplexes)
    simplexes = find_mins_AB_and_tolerance_for_simplexes(simplexes)
    pareto_front = get_pareto_front(points)

    previous_tolerance = None
    for i in xrange(max_iters):
        # Find simplex with lowest prognosed mins_AB (its compromise)
        simplex_to_divide = select_simplex_to_divide(simplexes)
        simplex_to_divide[-1]['hash'] = hash(str(simplex_to_divide[:-1]))
        # Test if tolerance has not increased
        if previous_tolerance is None:
            previous_tolerance = simplex_to_divide[-1]['tolerance']
        else:
            if round(previous_tolerance, 6) < round(simplex_to_divide[-1]['tolerance'], 6):
                print 'Tolerance test failed: %d. %f < %f' % (i, previous_tolerance, simplex_to_divide[-1]['tolerance'])
                import ipdb; ipdb.set_trace()
                # draw_simplex_3d_euclidean_bounds(simplex_to_divide, points=[[1.0, 0.125, 0.875], [1.0, 0.5, 1.0]])
                # get_simplex_analytical_lb_min(simplex_to_divide[:-1], [1.,1.])
                pass
            previous_tolerance = simplex_to_divide[-1]['tolerance']

        # Chose division point
        division_point = get_division_point(*simplex_to_divide[:2])

        # Update pareto front
        update_pareto_front(pareto_front, division_point)

        # Add new simplexes, remove divided one
        new_simplex1 = sort_vertexes_longest_edge_first([simplex_to_divide[0], division_point, simplex_to_divide[2]])
        new_simplex2 = sort_vertexes_longest_edge_first([simplex_to_divide[1], division_point, simplex_to_divide[2]])
        new_simplex1.append({'mins_AB': find_mins_AB(new_simplex1, L), 'tolerance': get_tolerance(new_simplex1, L),
                             'mins_ABC': find_mins_ABC(new_simplex1, L), 'approx_mins_ABC': get_approx_min_ABC(new_simplex1),
                             'parent_hash': hash(str(simplex_to_divide[:-1])), 'hash': hash(str(new_simplex1[:-1]))})
        new_simplex2.append({'mins_AB': find_mins_AB(new_simplex2, L), 'tolerance': get_tolerance(new_simplex2, L),
                             'mins_ABC': find_mins_ABC(new_simplex2, L), 'approx_mins_ABC': get_approx_min_ABC(new_simplex2),
                             'parent_hash': hash(str(simplex_to_divide[:-1])), 'hash': hash(str(new_simplex2[:-1]))})

        index = simplexes.index(simplex_to_divide)
        simplexes.remove(simplex_to_divide)
        simplexes.insert(index, new_simplex1)
        simplexes.insert(index, new_simplex2)

        # remove_dominated_simplexes(simplexes)

        print i, simplex_to_divide[-1]['tolerance']  #, # simplex_to_divide

    return pareto_front, simplexes



if __name__ == '__main__':
    ## Pavaizduoti testinius simpleksus ir išsiaiškinti, kaip apskaičiuojama tolerancija.

    ### simplex_pp = [[0.0, 0.0, {'obj': (0.0, 1.4142135623730949)}],
    ###      [2.0, 2.0, {'obj': (2.0, 1.7677669529663687)}],
    ###      [2.0, 0.0, {'obj': (2.0, 0.35355339059327373)}],
    ###     {'tolerance': 2.0,
    ###      'mins_AB': [a([ 0.29289322,  0.29289322, -0.41421356]), a([ 0.91161165,  0.91161165, -0.40900974])], 
    ###      'hash': -434798978226135347}]

    ### simplex_p = [[0.0, 0.0, {'obj': (0.0, 1.4142135623730949)}],
    ###       [2.0, 0.0, {'obj': (2.0, 0.35355339059327373)}],
    ###       [1.0, 1.0, {'obj': (1.0, 1.4142135623730949)}],
    ###      {'parent_hash': -434798978226135347,
    ###       'tolerance': 0.66421356237309515,
    ###       'mins_AB': [a([ 0.,  0.,  0.]), a([ 1.375,  0., -0.53033009])],
    ###       'hash': -5074806818602029973}]

    ### # simplex = [[0.0, 0.0, {'obj': (0.0, 1.4142135623730949)}],
    ### #       [1.0, 1.0, {'obj': (1.0, 1.4142135623730949)}],
    ### #       [1.0, 0.0, {'obj': (1.0, 0.70710678118654746)}],
    ### #      {'parent_hash': -5074806818602029973,
    ### #       'tolerance': 1.0,
    ### #       'mins_AB': [a([ 0.14644661,  0.14644661, -0.20710678]), a([ 0.5,  0.5,  0.41421356])], 
    ### #       'hash': -7635020306800345344}]

    # simplex1_p = [[2.0, 2.0, {'obj': (2.0, 1.7677669529663687)}],
    #              [0.0, 2.0, {'obj': (0.0, 2.8284271247461898)}],
    #              [1.0, 1.0, {'obj': (1.0, 1.4142135623730949)}],
    #              {'parent_hash': 2788087925405831476,
    #               'tolerance': 0.8214285670894067,
    #               'mins_AB': [a([ 0.,  2.,  0.]), a([ 1.375,  2.,  0.88388348])],
    #               'hash': -3509271369440614653,
    #               'mins_ABC': [[1.4610251371999295e-08, 1.9999999874248653, 1.4610251453106571e-08],
    #                            [1.5892857095626787, 1.5892857194077434, 1.1869292457032219]]}]

    # simplex1 = [[2.0, 2.0, {'obj': (2.0, 1.7677669529663687)}],
    #            [1.0, 1.0, {'obj': (1.0, 1.4142135623730949)}],
    #            [1.0, 2.0, {'obj': (1.0, 2.1213203435596424)}],
    #            {'parent_hash': -3509271369440614653, 'tolerance': 1.2907569017587937,
    #             'mins_AB': [a([ 1.14644661,  1.14644661,  0.79289322]),
    #                        a([ 1.41161165,  1.41161165,  0.59099026])],
    #             'hash': 7582923602373750142,
    #             'mins_ABC': [[1.0000000022790292, 1.2500000040405155, 0.75000000424753255],
    #                          [1.124999995869711, 1.1249999995096069, 1.2374368703437737]]}]

    simplex2_p = [[1.0, 1.0, {'obj': (1.0, 1.4142135623730949)}],
                  [1.0, 1.5, {'obj': (1.0, 1.7677669529663687)}],
                  [1.25, 1.25, {'obj': (1.25, 1.7677669529663687)}],
                  {'hash': -213673139274823480,
                   'mins_AB': [a([ 1.  ,  1.25,  0.75]),
                              a([ 1.  ,  1.125,  1.23743687])],
                   'mins_ABC': [[1.0000000039540604, 1.4375000038639874, 0.93750000386398724],
                                [1.0011349252908066, 1.0011349261054265, 1.4158185896876572]],
                   'parent_hash': -2682190357071842817,
                   'tolerance': 0.47066325329217262}]

    simplex2 = [[1.0, 1.0, {'obj': (1.0, 1.4142135623730949)}],
                [1.25, 1.25, {'obj': (1.25, 1.7677669529663687)}],
                [1.0, 1.25, {'obj': (1.0, 1.5909902576697319)}],
                {'hash': 1548586294113774436,
                 'mins_AB': [a([ 1.03661165,  1.03661165,  0.9482233 ]),
                            a([ 1.03661165,  1.03661165,  1.34099026])],
                 'mins_ABC': [[1.000000004926255, 1.0624999982060133, 0.93750000286461188],
                              [1.0000000034316754, 1.0000000471724735, 1.4142135981556303]],
                 'parent_hash': -213673139274823480,
                 'tolerance': 0.47186462349655484}]


    # simplex3_p = [[2.0, 0.0, {'obj': (2.0, 0.35355339059327373)}],
    #               [1.0, 0.0, {'obj': (1.0, 0.70710678118654746)}],
    #               [1.5, 0.5, {'obj': (1.5, 1.0606601717798212)}],
    #               {'hash': -5016353933918973602,
    #                'mins_AB': [a([ 1.,  0.,  1.]),
    #                           a([ 1.625,  0., -0.1767767])],
    #                'mins_ABC': [[1.1852583953226197, 0.1852583929555576, 1.164461518209515],
    #                             [1.9998399000505764, 0.00016009809500141094, 0.35377980480180915]],
    #                'parent_hash': 4450402591846051078,
    #                'tolerance': 1.0424188487684969}]

    # simplex3 = [[1.0, 0.0, {'obj': (1.0, 0.70710678118654746)}],
    #             [1.5, 0.5, {'obj': (1.5, 1.0606601717798212)}],
    #             [1.5, 0.0, {'obj': (1.5, 0.70710678118654746)}],
    #             {'hash': -9066259061307278331,
    #              'mins_AB': [a([ 1.0732233 ,  0.0732233 ,  0.89644661]),
    #                         a([ 1.16161165,  0.16161165,  0.38388348])],
    #              'mins_ABC': [[1.000003146660088, 1.2879233517033395e-09, 1.000003146660088],
    #                           [1.2196699166074148, 5.8604735108367654e-09, 0.48743687340961339]],
    #              'parent_hash': -5016353933918973602,
    #              'tolerance': 1.0732201517101196}]



    ## Tėvinio simplekso atveju, kuris p nulemia tolerancijos dydį
    # simplex_p.pop(-1)
    # simplex.pop(-1)
    # print 'simplex parent mins_ABC', get_tolerance(simplex_p, [1., 1.])
    # print 'simplex mins_ABC:', get_tolerance(simplex, [1., 1.])

    ## Dogmos:
    ## Vienos tikslo funkcijos apatinės ribos minimumas gali tik didėti bet kaip dalinant simpleksą.
    ## Veinam simpleksui turi būti grąžinamas tik vienas skaitinis įvertis nepriklausomai nuo įverčio apskaičiavimo strategijos
    ## Tolerancija dalinant turi nedidėti.

    ## Vizualizavimas:
    # draw_simplex_3d_euclidean_bounds(simplex_p)  # Vienos tikslo funkcijos apatinę ribą pavaizduoja
    # draw_two_objectives_for_2d_simplex(simplex2)  # Apatinės simpleksų ribos vaizdas
    ## show_partitioning([simplex_p, simplex]);  # Pavaizduoja simpleksus kintamųjų erdvėje
    # exit()

    ## Klausimas:
    # Nupiešti grafiką, kaip kinta tolerancija viršūnę judinant kuria nors
    #   kraštine (tolerancija apskaičiuojama 3D atveju)

    # Kas atsitinka, jeigu padalinama ne blogiausios tolerancijos taške.
    #   Ar tokiu atveju vaikinis simpleksas gali turėti didesnę toleranciją?

    # Realizuoti algoritmą vienos tikslo funkcijos atveju
    # Funkcijos reikšmė dabar yra padalinama iš jos Lipshitco konstantos - ar tai tinkama strategija?



    ### v1, v2 = simplex[0], simplex[1]
    ### dist = enorm(a(v1[:-1]) - a(v2[:-1]))
    ### y1 = v1[-1]['obj']
    ### y2 = v2[-1]['obj']
    ### x1 = [0.0, {'obj': v1[-1]['obj']}]
    ### x2 = [dist, {'obj': v2[-1]['obj']}]


    ### ## Ar mažinant kraštinės ilgį vienmačiu atveju, tolerancija būtinai mažėja?
    ### # Testinis simpleksas:

    ### # Parodyti, ar ilgesnė kraštinė gali turėti žemesnę toleranciją, nei trumpesnė
    ### # Teoriškai taip, o praktiškai ne. Todėl reikia sutvarkyti kodą, kad atitiktų teoriją.
    ### # Išsiaiškinti, kodėl braižoma tokia tolerancija?
    ### u = 1.9
    ### from utils import lower_bound
    ### sv1 = [0.0, 0, {'obj': v1[-1]['obj']}]
    ### y1, z1 = v1[-1]['obj']
    ### y2, z2 = v2[-1]['obj']
    ### # B = [A[0]+u, {'obj': [lower_bound(u,y1,y2,v), lower_bound(u,z1,z2,v)]}]
    ### # print 'got', lower_bound(dist,y1,y2,dist), 'expected', y2, dist,y1,y2,dist
    ### sv2 = [dist, 0, {'obj': [lower_bound(dist-u, y1, y2, dist), lower_bound(dist-u, z1, z2, dist)]}]
    ### sv3 = x1

    ### # tol = get_tolerance(simplex, [1.,1.])
    ### stub_simplex = [sv1, sv2, sv3, {}]
    ### tol = get_tolerance(stub_simplex, [1.,1.])
    ### print 'dist ' + str(dist) + ' tolerance ' + str(tol)
    ### # plt.title('dist ' + str(dist) + ' tolerance ' + str(tol))

    ### # '''Returns lower Lipschitz bound value at x1+u point.
    ### # o1, o2 - value at x1, x2.
    ### # v - interval length.'''
    ### # l_break = (1.5 + 0 - 2)/ 2 = -0.25   # l_break cannot be negative, because its between x1, x2   # Input: 1.5 0.0 2.0 1.5

    ### draw_bounds(sv1, sv2)
    # exit()

    # simplexes = [
    #     [[0.0, 0.0, {'obj': (0.0, 1.4142135623730949)}],
    #      [2.0, 2.0, {'obj': (2.0, 1.7677669529663687)}],
    #      [2.0, 0.0, {'obj': (2.0, 0.35355339059327373)}], {}],
    #     [[0.0, 0.0, {'obj': (0.0, 1.4142135623730949)}],
    #      [2.0, 0.0, {'obj': (2.0, 0.35355339059327373)}],
    #      [1.0, 1.0, {'obj': (1.0, 1.4142135623730949)}], {}]]

    max_iters = 500
    max_tolerance = 5 * 10**-3   # max epsilon
    L = [1., sqrt(2)]
    lb = [0., 0.]
    ub = [2., 2.]
    D = 2   # Number of dimensions

    ## Kaip galima sustojimo kriterijų apskaičiuoti?
    ##   Imti mins_AB ir žinomos reikšmės minimumo santykį.
    ## Kaip galima apjungti dvi tikslo funkcijas? Straipsnio metodiką panaudoti.
    ## Kaip galima palyginti metodų veikimo kokybę?
    pareto_front, simplexes = bb_angled_algorithm(lb, ub, L)

    show_simplex_minimum_dissmatch(simplexes)
    show_lower_pareto_bound(simplexes)
    show_partitioning(simplexes)
    show_pareto_front(pareto_front)
