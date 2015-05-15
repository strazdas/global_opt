#! coding: utf-8
'''
Dviejų kintamųjų, vienos tikslo funkcijos problemos sprendimo algoritmas, kurio
strategija dalinti stačiais trikmapiais, ilgiausią kraštinę dalinant pusiau.

Pradinė erdvė padalinama į keturis stačiuosius trikampius.
Toliaus dalinami trikampiai, kurių tolerancija mažiausia (apatinės ribos minimumas, padalintas iš cos alpha).
'''
from copy import copy
from itertools import permutations
from numpy import array as a, sqrt, isnan, isinf
from utils import enorm, city_block_norm, show_pareto_front, show_partitioning,\
                  draw_bounds, remove_dominated_subinterval, show_lower_pareto_bound,\
                  draw_simplex_3d_euclidean_bounds, draw_two_objectives_for_2d_simplex
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from numpy.linalg import det


# def f1(X):
#     '''First problem form multiobjective-bivariate algorithm paper 5 chapter'''
#     return X[0]
# def f2(X):
#     return (min(abs(X[0] - 1), 1.5 - X[0]) + X[1] + 1) / sqrt(2)

def f1(X):
    return np.sqrt((X[0]-1)**2 + (X[1]-1)**2)


def triangulate(lb, ub):
    '''Triangulates the given hyperrectangle using combinatoric triangulation.'''
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
        x.append({'obj': (f1(x),)})
        return x
    return [x, {'obj': (f1(x),)}]

def find_mins_AB_and_tolerance_for_simplexes(simplexes, L):
    '''Groups points into intervals and computes tolerance for each interval.'''
    for simplex in simplexes:
        sort_vertexes_longest_edge_first(simplex)
        if type(simplex[-1]) == dict:
            simplex[-1]['mins_AB'] = find_mins_AB(simplex, L)
            simplex[-1]['approx_min_ABC'] = get_simplex_approx_lb_min(simplex, L)
            simplex[-1]['tolerance'] = get_tolerance(simplex, L)
        else:
            simplex.append({
                'mins_AB': find_mins_AB(simplex, L),
                'approx_min_ABC': get_simplex_approx_lb_min(simplex, L),
                'tolerance': get_tolerance(simplex, L),
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
    Finds AB' and B'A intersection, where A, B are longest edge vertexes
    t - triangle (simplex).
    y - objective values for each vertex.
    Returns lower Lipschitz bound minimum for the first edge (made from first
    and second vetexes).
    '''
    mins_AB = []
    t = a([v[:-1] for v in simplex])
    y = a([v[-1]['obj'] for v in simplex])

    # For each objective find the minimum AB point
    for i in xrange(1):
        f = y[:,i]    # function values
        l = L
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

def get_simplex_approx_lb_min(simplex, L):
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
    return min([   # Simpleksą aproksimuojantį minimumą turim
        v1[-1]['obj'][0] - L*enorm(v1[:2] - a(mins_AB[0][:2]))/np.cos(get_alpha(C-A, B-A)),
        v2[-1]['obj'][0] - L*enorm(v2[:2] - a(mins_AB[0][:2]))/np.cos(get_alpha(C-B, A-B))
    ])


# def get_AB_cos_tolerance(simplex):
#     '''Formulė šiai tolerancijai apskaičiuoti:
#     dist_A = ||A[:2] - min_AB[:2]|| / fiA
#     dist_B = ||B[:2] - min_AB[:2]|| / fiB
# 
#     min(y1 - dist_A*L, y2 - dist_B*L)
# 
#     Jeigu lygūs, tai po padidinimo reikia imti aukštensį.
#     '''
#     # Note: this tolerance definition is wrong, because we get tolygus leistinosios srities padengimas
#     if type(simplex[-1]) == dict and simplex[-1].has_key['approx_min_ABC']:
#         approx_min_ABC = simplex[-1]['approx_min_ABC']
#     else:
#         approx_min_ABC = get_simplex_approx_lb_min(simplex, L)
#     return max([simplex[0][-1]['obj'][0] - approx_min_ABC,
#                 simplex[1][-1]['obj'][0] - approx_min_ABC])


def get_lb_min_tolerance(simplex):
    if type(simplex[-1]) == dict and simplex[-1].has_key['approx_min_ABC']:
        approx_min_ABC = simplex[-1]['approx_min_ABC']
    else:
        approx_min_ABC = get_simplex_approx_lb_min(simplex, L)
    return approx_min_ABC


def get_tolerance(simplex, L):
    ## Two objective tolerances:
    # return get_max_vertex_obj_dist_to_obj_lb_min_points(simplex, L)
    # return get_max_vertex_obj_dist_to_obj_lb_mins(simplex, L)
    # get_max_vertex_obj_city_block_dist_to_obj_lb_mins(simplex, L)
    ## One objective tolerance:
    # return get_lb_min_tolerance(simplex)
    # return get_AB_cos_tolerance(simplex)
    # return get_simplex_approx_lb_min(simplex, L)
    return None


def select_simplex_to_divide(simplexes):
    # return max(simplexes, key=lambda x: x[-1]['tolerance'])
    return min(simplexes, key=lambda x: x[-1]['approx_min_ABC'])

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


def bu_angled_algorithm(lb, ub, L):
    # Pastaba: Padaliną leistinąją sritį į keturis stačiuosius trikampius (jeigu leistinoji sritis keturkampis)
    simplexes = triangulate(lb, ub)
    points = find_objective_values_for_vertexes(simplexes)
    simplexes = find_mins_AB_and_tolerance_for_simplexes(simplexes, L)
    pareto_front = get_pareto_front(points)

    # previous_tolerance = None
    for i in xrange(max_iters):
        # Find simplex with lowest prognosed mins_AB (its compromise)
        simplex_to_divide = select_simplex_to_divide(simplexes)
        simplex_to_divide[-1]['hash'] = hash(str(simplex_to_divide[:-1]))
        ## Test if tolerance has not increased

        # if previous_tolerance is None:
        #     previous_tolerance = simplex_to_divide[-1]['tolerance']
        # else:
        #     if round(previous_tolerance, 6) < round(simplex_to_divide[-1]['tolerance'], 6):
        #         print 'Tolerance test failed: %d. %f < %f' % (i, previous_tolerance, simplex_to_divide[-1]['tolerance'])
        #         import ipdb; ipdb.set_trace()
        #         pass
        #     previous_tolerance = simplex_to_divide[-1]['tolerance']

        # Chose division point
        division_point = get_division_point(*simplex_to_divide[:2])

        # Update pareto front
        update_pareto_front(pareto_front, division_point)

        # Add new simplexes, remove divided one
        new_simplex1 = sort_vertexes_longest_edge_first([simplex_to_divide[0], division_point, simplex_to_divide[2]])
        new_simplex2 = sort_vertexes_longest_edge_first([simplex_to_divide[1], division_point, simplex_to_divide[2]])
        new_simplex1.append({'mins_AB': find_mins_AB(new_simplex1, L), 'tolerance': get_tolerance(new_simplex1, L),
                             'mins_ABC': find_mins_ABC(new_simplex1, L), 'approx_min_ABC': get_simplex_approx_lb_min(new_simplex1, L),
                             'parent_hash': hash(str(simplex_to_divide[:-1])), 'hash': hash(str(new_simplex1[:-1]))})
        new_simplex2.append({'mins_AB': find_mins_AB(new_simplex2, L), 'tolerance': get_tolerance(new_simplex2, L),
                             'mins_ABC': find_mins_ABC(new_simplex2, L), 'approx_min_ABC': get_simplex_approx_lb_min(new_simplex2, L),
                             'parent_hash': hash(str(simplex_to_divide[:-1])), 'hash': hash(str(new_simplex2[:-1]))})

        index = simplexes.index(simplex_to_divide)
        simplexes.remove(simplex_to_divide)
        simplexes.insert(index, new_simplex1)
        simplexes.insert(index, new_simplex2)

        remove_dominated_simplexes(simplexes)
        if i == 500:
            show_partitioning(simplexes)
            exit()

        # print i, simplex_to_divide[-1]['tolerance']#, # simplex_to_divide
        print i, simplex_to_divide[-1]['approx_min_ABC']#, # simplex_to_divide

    return pareto_front, simplexes


if __name__ == '__main__':
    ## Dogmos:
    ## Vienos tikslo funkcijos apatinės ribos minimumas gali tik didėti bet kaip dalinant simpleksą.
    ## Vienam simpleksui turi būti grąžinamas tik vienas skaitinis įvertis nepriklausomai nuo įverčio apskaičiavimo strategijos
    ## Tolerancija dalinant turi nedidėti.

    max_iters = 500
    max_tolerance = 5 * 10**-3   # max epsilon
    L = 4.

    lb = [0., 0.]
    ub = [2., 2.]
    D = 2   # Number of dimensions

    pareto_front, simplexes = bu_angled_algorithm(lb, ub, L)

    # show_lower_pareto_bound(simplexes)
    show_partitioning(simplexes)
    show_pareto_front(pareto_front)
