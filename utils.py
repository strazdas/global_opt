# coding: utf-8
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from numpy import sqrt
from numpy import array as a, matrix as m
from numpy import array as a, matrix as m, arange, sqrt, isnan, pi, cos, sin, mean
from matplotlib import pyplot as plt
from random import random, seed
from numpy.linalg import det

import matplotlib.ticker as plticker
from matplotlib import cm


def enorm(X):
    '''Euclidean norm'''
    if isinstance(X, (int, float)):  # len(X) < 2:
        return abs(X)
    return sqrt(sum([e**2 for e in X]))

def l2norm(a1, a2):
    '''Euclidean norm, which converts arguments to arrays automatically.'''
    if isinstance(a1, (int, float)):  # len(X) < 2:
        return abs(a(a1)-a(a2))
    return sqrt(sum([e**2 for e in (a(a1)-a(a2))]))

def nm(obj):
    '''No meta - returns object without meta iformation.'''
    if type(obj[-1]) == dict:
        return obj[:-1]
    return obj

def city_block_norm(X):
    if len(X) < 2:
        return X
    return sum([e for e in X])


# def lower_bound(p, t, y, norm='Euclidean', v_num=2, L=[1, 1]):
#     '''p - point at which the lower L bound is searched.
#     t - triangle vertex coordinates.
#     y - f1, f2 values at triangle vertexes.
#     norm - which type of norm to use.
#     v_num - number of vertexes to use for approximation.
#     '''
#     # Warning:  works not like expected
#     # 2.32842712475 2.0 2.5 2.82842712475
#     # 2.32842712475 0.0 2.0 2.82842712475
#     #   y1 = max([f[0] - L[0]*city_block_norm(v - a(p)) for v, f in zip(t, y)[:v_num]])
#     #   TypeError: zip argument #1 must support iteration]
#
#     t = a(t)
#     y = a(y)
#     if norm == 'Euclidean':
#         # reikšmė - L * atstumas
#         y1 = max([f[0] - L[0]*enorm(v - a(p)) for v, f in zip(t, y)[:v_num]])
#         y2 = max([f[1] - L[1]*enorm(v - a(p)) for v, f in zip(t, y)[:v_num]])
#         return [y1, y2]
#     y1 = max([f[0] - L[0]*city_block_norm(v - a(p)) for v, f in zip(t, y)[:v_num]])
#     y2 = max([f[1] - L[1]*city_block_norm(v - a(p)) for v, f in zip(t, y)[:v_num]])
#     return [y1, y2]


#########  Algorithm result visualization utilities  #########
def show_pareto_front(pareto_front):
    '''Draws 2D pareto set and 2D pareto front.'''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    obj1 = [p[-1]['obj'][0] for p in pareto_front]
    obj2 = [p[-1]['obj'][1] for p in pareto_front]
    ax2.plot(obj1, obj2, 'bo')
    ax1.plot([p[0] for p in pareto_front], [p[1] for p in pareto_front], 'ro')
    plt.show()


def show_partitioning(simplexes, simplex_to_divide=None, division_point=None):
    nr_of_colors = plt.cm.jet.N / len(simplexes)
    clrs = plt.cm.jet([e*nr_of_colors for e in range(len(simplexes))])
    for i, simplex in enumerate(simplexes):
        s = simplex[:-1]
        # simplex_center = [mean([v[0] for v in simplex[:-1]]), mean([v[1] for v in simplex[:-1]])]
        # plt.text(simplex_center[0], simplex_center[1], '%.4f' % simplex[-1]['tolerance'])
        for j in range(3):
            plt.plot([s[j-1][0], s[j][0]], [s[j-1][1], s[j][1]], '-', color=clrs[i])

    # Color in other color newly added edge color
    if simplex_to_divide:
        v1 = simplex_to_divide[2]
        v2 = division_point
        plt.plot([v1[0], v2[0]], [v1[1], v2[1]], 'r-')

    plt.show()



def show_lower_pareto_bound(simplexes):
    '''For 2D -> 2D problem show the lower pareto bound.'''
    # Warning:  unfinished, untested.
    def lower_bound_for_interval(t, y, dist=None, L=[1.,1.], verts=[0,1]):
        '''Returns lower L bound line in (f1,f2) space for an interval in
        multidimensional feasible region.'''
        def lower_bound_cracks(x1, x2, y1, y2, dist):
            '''Computes lower L bound crack points for the given interval.'''
            if dist is None:
                dist = enorm(x2-x1)
            t1 = (y1[0]-y2[0])/(2.*L[0]) + dist/2.
            t2 = (y1[1]-y2[1])/(2.*L[1]) + dist/2.
            if t1 >= t2:
                p1 = [y1[0] - L[0]*t2, y1[1] - L[1]*t2]
                p2 = [y1[0] - L[0]*t1, y2[1] - dist*L[1] + L[1]*t1]
            else:
                p2 = [y2[0] - dist*L[0] + L[0]*t2, y1[1] - L[1]*t2]
                p1 = [y1[0] - L[0]*t1, y1[1] - L[1]*t1]
            return [p1, p2]
        p1, p2 = lower_bound_cracks(t[verts[0]], t[verts[1]], y[verts[0]], y[verts[1]], dist)
        return a([y[verts[0]], p1, p2, y[verts[1]]])

    # For each simplex we have dimensions+1 vertex, so what does this mean.
    # Patobulinimas: We could check if any of them are dominated and keep the
    # least dominated.
    for simplex in simplexes:
        t = a([simplex[0][:-1], simplex[1][:-1], simplex[2][:-1]])
        y = a([simplex[0][-1]['obj'], simplex[1][-1]['obj'], simplex[2][-1]['obj']])   # (obj1, obj2) for A, B, C
        lb = lower_bound_for_interval(t, y)
        plt.plot(lb[:,0], lb[:,1])

    # Draw longest (or nondominated) edge lower bound and mark these vertexes as stars.
    plt.show()


def show_simplex_minimum_dissmatch(simplexes):
    '''Shows approximated minimum dismatch with analytically found solution.'''
    min_dissmatches = []
    for simplex in simplexes:
        approx_mins_ABC = simplex[-1]['approx_mins_ABC']
        mins_ABC = [o[-1] for o in simplex[-1]['mins_ABC']]

        dist_approx_min_ABC = max([enorm(a(v[-1]['obj']) - a(approx_mins_ABC)) for v in simplex[:-1]])
        dist_min_ABC = max([enorm(a(v[-1]['obj']) - a(mins_ABC)) for v in simplex[:-1]])

        min_dissmatch = dist_approx_min_ABC / dist_min_ABC
        min_dissmatches.append(min_dissmatch)
    plt.plot(range(len(min_dissmatches)), min_dissmatches)
    plt.title('Approximated lower bound minimum dissmatch with the real lower bound minimum')

    # Draw mean line
    avg_dissmatch = np.mean(min_dissmatches)
    plt.plot([0, len(min_dissmatches)-1], [avg_dissmatch, avg_dissmatch], 'm--')
    plt.show()

def draw_3d_objective_function(f, lb=[-0.1, 2.1], ub=[-0.1, 2.1], title=''):
    X1 =  np.arange(lb[0], ub[0], (ub[0]-lb[0])/150.)
    X2 =  np.arange(lb[1], ub[1], (ub[1]-lb[1])/150.)
    X1, X2 = np.meshgrid(X1, X2)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    X3 = []
    for i, x1_row in enumerate(X1):
        row = []
        for j, x1 in enumerate(x1_row):
            row.append(f([x1, X2[i,j]]))
        X3.append(row)

    ax.plot_surface(X1, X2, X3, linewidth=0, rstride=1, cstride=1, cmap=cm.gist_rainbow) #jet) # cm.coolwarm)
    ax.set_xlim(lb[0], ub[0])
    ax.set_ylim(lb[1], ub[1])
    # ax.set_zlim(lb[2], ub[2])
    ax.set_title(title)
    plt.show()


#########  Simplex lower bound visualisation utilities  #########
def get_tolerance(x1, x2, L=[1.,1.]):
    '''Returns tolerance for the interval for 1D->2D problem.

    Warning this functions does not support Lipschitz constant.
    '''
    y1 = x1[-1]['obj']
    y2 = x2[-1]['obj']
    # FIXME: check if lower bound calculation is correct
    def lower_bound_cracks(x1, x2, y1, y2, dist=None):
        '''Computes lower L bound crack points for the given interval.'''
        if dist is None:
            dist = enorm(x2-x1)
        t1 = (y1[0]-y2[0])/(2.*L[0]) + dist/2.
        t2 = (y1[1]-y2[1])/(2.*L[1]) + dist/2.
        if t1 >= t2:
            p1 = [y1[0] - L[0]*t2, y1[1] - L[1]*t2]
            p2 = [y1[0] - L[0]*t1, y2[1] - dist*L[1] + L[1]*t1]
        else:
            p1 = [y1[0] - L[0]*t1, y1[1] - L[1]*t1]
            p2 = [y2[0] - dist*L[0] + L[0]*t2, y1[1] - L[1]*t2]
        return [p1, p2]
    p1, p2 = lower_bound_cracks(x1[0], x2[0], y1, y2)
    return max([min((enorm(a(p1) - a(y1)), enorm(a(p2) - a(y1)))),
                min((enorm(a(p1) - a(y2)), enorm(a(p2) - a(y2))))])

def U(u, o1, o2, v):
    '''Returns upper U bound value at x1 + u point.
    u - distance from x1 moving towards x2
    o1, o2 - value at x1, x2.
    v - interval length.'''
    if o2 > o1:
        if u > abs(o2 - o1):
            return o2
        return o1 + u
    if u > v - abs(o2 - o1):
        return o2 + (v - u)
    return o1

def lower_bound(u, o1, o2, v):
    '''Returns lower Lipschitz bound value at x1+u point.
    o1, o2 - value at x1, x2.
    v - interval length.'''
    l_break = (v + o1 - o2) / 2.
    if u < l_break:
        return o1 - u
    return o1 - l_break + (u - l_break)

def remove_dominated_subinterval(A, B):
    '''Single variable bi-dimensional dominated subinterval removal'''
    y1, z1 = A[-1]['obj']
    y2, z2 = B[-1]['obj']

    if (y1 >= y2 and z2 >= z1) or (y2 >= y1 and z1 >= z2):   # If strick domination is used, then all the interval is removed
        return A, B                                          # Cannot use strick domination definition in this case.

    v = abs(A[0] - B[0])
    y_break = (v + y1 - y2) / 2.
    z_break = (v + z1 - z2) / 2.
    if y1 <= y2 and z1 <= z2:
        u = max([2*y_break, 2*z_break])
        B = [A[0]+u, {'obj': [lower_bound(u,y1,y2,v), lower_bound(u,z1,z2,v)]}]
    elif y2 <= y1 and z2 <= z1:
        u = min([v - 2*(v - y_break), v - 2*(v-z_break)])
        A = [A[0]+u, {'obj': [lower_bound(u,y1,y2,v), lower_bound(u,z1,z2,v)]}]
    return A, B


def get_Lipschitz_bound(x1, x2, t='lower', L=[1.,1.]):
    '''Finds lower and upper Lipschitz bounds for 1D->2D problem to be drawn in objectives space.'''
    v = x2[0] - x1[0]
    y1, z1 = x1[-1]['obj']
    y2, z2 = x2[-1]['obj']
    s = -1 if t == 'lower' else 1

    y_break = (v - s*(y1 - y2)/L[0]) / 2.
    z_break = (v - s*(z1 - z2)/L[1]) / 2.

    l_break = min([y_break, z_break])
    h_break = max([y_break, z_break])

    is_y_h = [y_break, z_break].index(l_break)
    if is_y_h:
        return a([(y1, z1), (y1+s*l_break*L[0], z1+s*l_break*L[1]), (y1+s*h_break*L[0], z1+s*2*l_break*L[1]-s*h_break*L[1]), (y2,z2)])
    return a([(y1, z1), (y1+s*l_break*L[0], z1+s*l_break*L[1]), (y1+s*2*l_break*L[0] - s*h_break*L[0], z1+s*h_break*L[1]), (y2, z2)])

def get_U_bound(x1, x2):
    '''Finds upper U Lipschitz bounds for 1D->2D problem to be drawn in objectives space.'''
    v = x2[0] - x1[0]
    y1, z1 = x1[-1]['obj']
    y2, z2 = x2[-1]['obj']
    yh = max([y1, y2])
    zh = max([z1, z2])
    z_U_break = abs(z2 - z1)
    if z1 > z2:
        z_U_break = v - z_U_break
    y_U_break = abs(y2 - y1)
    if y1 > y2:
        y_U_break = v - y_U_break
    if y_U_break < z_U_break:
        return a([(y1,z1), (yh,U(y_U_break,z1,z2,v)), (U(z_U_break,y1,y2,v),zh), (y2,z2)])
    return a([(y1,z1), (U(z_U_break,y1,y2,v),zh), (yh,U(y_U_break,z1,z2,v)), (y2,z2)])

def draw_objective_bounds_and_hat_epsilon(x1, x2, ax, L=[1.,1.]):
    '''Draws objective bounds in 1D variable x 1D objective plot.
    Plots overlap for each objective.'''
    v = x2[0] - x1[0]
    y1, z1 = x1[-1]['obj']
    y2, z2 = x2[-1]['obj']
    yh = max([y1, y2])
    zh = max([z1, z2])
    ## Draw first objective Lipschitz lower bound
    lower_break1 = (v + (y1 - y2)/L[0]) / 2.
    ax.plot([x1[0], x1[0]+lower_break1, x2[0]], [y1, y1 - L[0]*lower_break1, y2], 'r')
    ## Draw second objective Lipschitz lower bound
    lower_break2 = (v + (z1 - z2)/L[1]) / 2.
    ax.plot([x1[0], x1[0]+lower_break2, x2[0]], [z1, z1 - L[1]*lower_break2, z2], 'r')
    ## Draw first objective Lipschitz upper bound
    upper_break1 = (v + (y2 - y1)/L[0]) / 2.
    ax.plot([x1[0], x1[0]+upper_break1, x2[0]], [y1, y1 + L[0]*upper_break1, y2], 'k')
    ## Draw second objective Lipschitz upper bound
    upper_break2 = (v + (z2 - z1)/L[1]) / 2.
    ax.plot([x1[0], x1[0]+upper_break2, x2[0]], [z1, z1 + L[1]*upper_break2, z2], 'k')
    ## Draw first objective U upper bound
    y_U_break = abs(y2 - y1)/L[0]
    if y1 > y2:
        y_U_break = v - y_U_break
    ax.plot([x1[0], x1[0]+y_U_break, x2[0]], [y1, yh, y2], 'g--', linewidth=2)
    ## Draw second objective U upper bound
    z_U_break = abs(z2 - z1)/L[1]
    if z1 > z2:
        z_U_break = v - z_U_break
    ax.plot([x1[0], x1[0]+z_U_break, x2[0]], [z1, zh, z2], 'b--', linewidth=2)

    # Plot starting point colors
    ax.plot([x1[0], x1[0]], [y1, z1], 'yo')
    ax.plot([x2[0], x2[0]], [y2, z2], 'go')

    ## Division point visualization
    # u = get_division_point_for_nondominated_points(x1, x2)
    # ax.plot([u+x1[0], u+x1[0]], [U(u, y1, y2,v), U(u,z1,z2,v)], 'r*', ms=10)

    ax.grid(True)
    return ax

def draw_tolerance_change(x1, x2, ax, L=[1.,1.]):
    '''Draws tolerance vs x for bouth subintervals, got by dividing the
    interval at u point.'''
    v = x2[0] - x1[0]
    y1, z1 = x1[-1]['obj']
    y2, z2 = x2[-1]['obj']
    ## Divide interval into two parts
    u_real = arange(x1[0], x2[0], v/200.)

    ## Draw each part maximum tolerance
    u_real_points = [[u, {'obj': [U(u-x1[0], y1, y2,v), U(u-x1[0],z1,z2,v)]}] for u in u_real]
    ax.plot(u_real, [get_tolerance(x1, u, L) for u in u_real_points], 'k--', linewidth=2)
    ax.plot(u_real, [get_tolerance(u, x2, L) for u in u_real_points], 'm--', linewidth=2)
    ax.grid(True)
    return ax

def draw_objective_bounds(x1, x2, ax, L=[1.,1.]):
    '''Draws objective bounds in 2D objectives space for univariate problem.'''
    v = x2[0] - x1[0]
    y1, z1 = x1[-1]['obj']
    y2, z2 = x2[-1]['obj']

    ax.plot([y1], [z1], 'yo')
    ax.plot([y2], [z2], 'go')

    # Draw lower bound for bouth objectives
    lb = get_Lipschitz_bound(x1, x2, 'lower', L=L)
    ax.plot(lb[:,0], lb[:,1], 'r-')

    # Draw upper Lipschitz bound for bouth objectives
    ub = get_Lipschitz_bound(x1, x2, 'upper', L=L)
    ax.plot(ub[:,0], ub[:,1], 'k-')

    # Draw upper U bound for bouth objectives
    uUb = get_U_bound(x1, x2)
    ax.plot(uUb[:,0], uUb[:,1], 'y--', linewidth=2)

    ## Division point visualization
    # u = get_division_point_for_nondominated_points(x1, x2, L)
    # ax.plot([U(u, y1, y2,v)], [U(u,z1,z2,v)], 'r*', ms=10)

    ax.grid(True)
    return ax


def draw_bounds(x1, x2, L=[1.,1.]):
    '''Draws 3 plots describing bounds for 1D->2D problems'''
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18,6))

    ## Normalize x1, x2 for 2D variable space
    x2 = [enorm(a(x2[:-1]) - a(x1[:-1])), x2[-1]]
    x1 = [0, x1[-1]]

    ax1 = draw_objective_bounds_and_hat_epsilon(x1, x2, ax=ax1, L=L)
    ax2 = draw_tolerance_change(x1, x2, ax=ax2, L=L)
    ax3 = draw_objective_bounds(x1, x2, ax=ax3, L=L)

    ax1.xaxis.set_major_locator(plticker.MultipleLocator(base=0.2))
    ax1.yaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
    # ax1.axis([0,1.,0,1.2])

    ax2.xaxis.set_major_locator(plticker.MultipleLocator(base=0.2))
    ax2.yaxis.set_major_locator(plticker.MultipleLocator(base=0.1))

    ax3.xaxis.set_major_locator(plticker.MultipleLocator(base=0.2))
    ax3.yaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
    # ax3.axis([0,1.2,0,1.2])

    plt.show()



###########   3D simplex lower bound visualization utilities   ##########
# step = 0.1

def lower_bound_surface(X1, X2, t, y, L):
    '''
    X1, X2 - coordinate ticks in x1,x2 plane.
    t - triangle vertex coordinates in x1,x2 plane.
    y - objective function values at vertexes.

    Takes first two vertexes as longest triangle edge and creates its lower bound surface.
    '''
    LB = []
    for i, x1_row in enumerate(X1):
        row = []
        for j, x1 in enumerate(x1_row):
            row.append(point_lower_bound([x1, X2[i,j]], t, y))
        LB.append(row)
    return a(LB)


def draw_simplex_3d_euclidean_bounds(simplex, obj_nr=0, L=[1., 1.], points=[]):
    '''2d->2d simplex lower bound surface.
    Draws a single objective below 2D simplex'''
    t = a([simplex[0][:-1], simplex[1][:-1], simplex[2][:-1]])
    y = a([simplex[0][-1]['obj'], simplex[1][-1]['obj'], simplex[2][-1]['obj']])   # (obj1, obj2) for A, B, C

    X1 =  np.arange(-0.1, 2.1, 0.05)
    X2 =  np.arange(-0.1, 2.1, 0.05)
    X1, X2 = np.meshgrid(X1, X2)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    LB = lower_bound_surface(X1, X2, t, y, L)

    ax.plot_surface(X1, X2, LB[:,:,obj_nr], linewidth=0, rstride=1, cstride=1, cmap=cm.coolwarm)
    ax.plot_wireframe(np.hstack((t[:,0], t[0,0])), np.hstack(([t[:,1], t[0,1]])), np.hstack((y[:,obj_nr],y[0,obj_nr])), zorder=1000)  ## Line surface

    # points = [simplex[-1]['mins_ABC'][1]]
    for p in points:
        ax.plot([p[0]], [p[1]], [p[2]], 'go')

    # points = [[1.5892857095626787, 1.5892857194077434, 1.186929245703222]]
    # for p in points:
    #     ax.plot([p[0]], [p[1]], [p[2]], 'ro')
    plt.show()


def point_lower_bound(p, t, y, norm='Euclidean', v_num=3):
    '''p - point at which the lower L bound is searched.
    t - triangle vertex coordinates.
    y - f1, f2 values at triangle vertexes.
    norm - which type of norm to use.
    v_num - number of vertexes to use for approximation.
    '''
    L = [1., 1.]
    t = a(t)
    y = a(y)
    if norm == 'Euclidean':
        y1 = max([f[0] - L[0]*enorm(v - a(p)) for v, f in zip(t, y)[:v_num]])
        y2 = max([f[1] - L[1]*enorm(v - a(p)) for v, f in zip(t, y)[:v_num]])
        return [y1, y2]
    y1 = max([f[0] - L[0]*city_block_norm(v - a(p)) for v, f in zip(t, y)[:v_num]])
    y2 = max([f[1] - L[1]*city_block_norm(v - a(p)) for v, f in zip(t, y)[:v_num]])
    return [y1, y2]


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

def draw_two_objectives_for_2d_simplex(simplex, add_points=[]):
    '''2d->2d simplex objective values.
    points - additional points to be displayed in the drawing.
    Draws two objectives of two variable space'''

    def gen_points(triangle, n=10000):
        seed(123)
        points = []
        while len(points) < n:
            point = [2*random(), 2*random()]
            if is_in_region(point, a(triangle)):
                points.append(point)
        return points

    def get_y(points, t, y):
        y_points = []
        for p in points:
            y_points.append(point_lower_bound(p, t, y))
        return y_points


    def lower_bound_for_interval(t, y, dist=None, L=[1.,1.], verts=[0,1]):
        '''Returns lower L bound line in (f1,f2) space for an interval in
        multidimensional feasible region.'''
        def lower_bound_cracks(x1, x2, y1, y2, dist):
            '''Computes lower L bound crack points for the given interval.'''
            if dist is None:
                dist = enorm(x2-x1)
            t1 = (y1[0]-y2[0])/(2.*L[0]) + dist/2.
            t2 = (y1[1]-y2[1])/(2.*L[1]) + dist/2.
            if t1 >= t2:
                p1 = [y1[0] - L[0]*t2, y1[1] - L[1]*t2]
                p2 = [y1[0] - L[0]*t1, y2[1] - dist*L[1] + L[1]*t1]
            else:
                p2 = [y2[0] - dist*L[0] + L[0]*t2, y1[1] - L[1]*t2]
                p1 = [y1[0] - L[0]*t1, y1[1] - L[1]*t1]
            return [p1, p2]
        p1, p2 = lower_bound_cracks(t[verts[0]], t[verts[1]], y[verts[0]], y[verts[1]], dist)
        return a([y[verts[0]], p1, p2, y[verts[1]]])

    def draw_points(points, labels=['x', 'y'], color='', label=''):
        x = a(points)
        handle = plt.plot(x[:,0], x[:,1], color+'o', label=label)
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
        return handle

    t = a([simplex[0][:-1], simplex[1][:-1], simplex[2][:-1]])
    y = a([simplex[0][-1]['obj'], simplex[1][-1]['obj'], simplex[2][-1]['obj']])   # (obj1, obj2) for A, B, C
    points = gen_points(simplex)
    y_points = get_y(points, t, y)

    draw_points(y_points, ['y$_1$', 'y$_2$'], 'r', u'ABC simplex lower bound')

    l_AB = lower_bound_for_interval(t, y, verts=[0,1])
    plt.plot(l_AB[:,0], l_AB[:,1], 'b-', label=u'AB lower bound')

    l_AC = lower_bound_for_interval(t, y, verts=[0,2])
    plt.plot(l_AC[:,0], l_AC[:,1], 'g-', label=u'AC lower bound')

    l_CB = lower_bound_for_interval(t, y, verts=[1,2])
    plt.plot(l_CB[:,0], l_CB[:,1], 'm-', label=u'CB lower bound')

    for min_point in simplex[-1]['mins_ABC']:
        objs = point_lower_bound(min_point[:2], t, y)
        plt.plot([objs[0]], [objs[1]], '*y', zorder=10, markersize=10)

    for p in add_points:
        if len(p) > 2:
            p = point_lower_bound(p[:2], t, y, norm='Euclidean', v_num=3)
            print p
        plt.plot([p[0]], [p[1]], 'go', zorder=100)

    plt.legend(loc=4)
    plt.show()
