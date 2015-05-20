from numpy import array as a
import numpy as np
from numpy import sin, cos, e, pi

# from nu_angled import nu_angled_algorithm
from disimpl_2v import disimpl_2v
# from utils import draw_3d_objective_function, show_partitioning
from datetime import datetime
from scipy.optimize import minimize


############   Objective functions   ############
def hyperparabola(X):  # n->1
    '''Hyper-parabola with minimum at [1.]*n.'''
    return (a(X)).dot(a(X))

def rosenbrock(X):
    '''https://en.wikipedia.org/wiki/Rosenbrock_function'''
    return sum([100 * (X[i+1] - X[i]**2)**2 + (X[i]-1)**2 for i in range(len(X)-1)])

def styblinski(X):
    '''Styblinski-Tang function
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    '''
    return sum([x**4 - 16*x**2 + 5*x for x in X]) / 2.

def branin(X):
    '''2D http://www.sfu.ca/~ssurjano/branin.html'''
    x1 = X[0]
    x2 = X[1]
    b = 5.1/(4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8*np.pi)
    return (x2 - b*x1**2 + c*x1 - r)**2 + s*(1 - t) * np.cos(x1) + s
    # return (y - (5.1/(4*pi^2))*x^2 + (5/pi)*x - 6)^2 + 10*(1 - 1/(8*pi)) * cos(x) + 10

def goldstein_price(X):
    '''http://www.sfu.ca/~ssurjano/goldpr.html'''
    x1 = X[0]
    x2 = X[1]
    part1 = 1 +(x1 + x2 +1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
    part2 = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
    return part1 * part2

def six_hump_camel_back(X):
    '''http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=44040
       http://www.sfu.ca/~ssurjano/camel6.html'''
    x1, x2 = X
    return (4 - 2.1*x1**2 + x1**4 / 3.) * x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2

def shubert(X):
    '''http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=44040
    http://www.sfu.ca/~ssurjano/shubert.html'''
    sum1 = 0
    sum2 = 0
    for i in range(1, 6):
        new1 = i * np.cos((i+1)*X[0]+i)
        new2 = i * np.cos((i+1)*X[1]+i)
        sum1 += new1
        sum2 += new2
    return sum1 * sum2

def alolyan(X):
    '''http://link.springer.com/article/10.1007%2Fs10898-012-0020-3'''
    x1, x2 = X
    return x1*x2**2 + x2*x1**2 - x1**3 - x2**3

def easom(X):
    '''http://link.springer.com/article/10.1007%2Fs10898-012-0020-3'''
    x1, x2 = X
    return -np.cos(x1) * np.cos(x2) * np.e**(-((x1-np.pi)**2 + (x2 -np.pi)**2))

def rastrigin(X, A=10):
    '''http://link.springer.com/article/10.1007%2Fs10898-012-0020-3'''
    x1, x2 = X
    return 2*A + x1**2 + x2**2 - A*(np.cos(2*np.pi*x1) + np.cos(2*np.pi *x2))

def hartman3(X):
    '''http://www.sfu.ca/~ssurjano/hart3.html'''
    alpha = (1.0, 1.2, 3.0, 3.2)
    A = a([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
    P = a([[0.3689, 0.1170, 0.2673],
           [0.4699, 0.4387, 0.7470],
           [0.1091, 0.8732, 0.5547],
           [0.0381, 0.5743, 0.8828]])
    sum1 = 0
    for i in range(4):
        sum2 = 0
        for j in range(3):
            sum2 += A[i][j] * (X[j] - P[i][j])**2
        sum1 += alpha[i] * np.exp(-sum2)
    return -sum1

def shekel5(X):
    '''http://www.sfu.ca/~ssurjano/shekel.html
    http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/Page2354.htm
    '''
    m = 5
    beta = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    C = a([[4.,1.,8.,6.,3.,2.,5.,8.,6.,7.],
           [4.,1.,8.,6.,7.,9.,5.,1.,2.,3.6],
           [4.,1.,8.,6.,3.,2.,3.,8.,6.,7.],
           [4.,1.,8.,6.,7.,9.,3.,1.,2.,3.6]])
    sum1 = 0
    for i in range(m):
        sum2 = 0
        for j in range(4):
            sum2 += (X[j] - C[j][i])**2
        sum1 += 1./(sum2 + beta[i])
    return - sum1

def shekel7(X):
    '''http://www.sfu.ca/~ssurjano/shekel.html'''
    m = 7
    beta = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    C = a([[4.,1.,8.,6.,3.,2.,5.,8.,6.,7.],
           [4.,1.,8.,6.,7.,9.,5.,1.,2.,3.6],
           [4.,1.,8.,6.,3.,2.,3.,8.,6.,7.],
           [4.,1.,8.,6.,7.,9.,3.,1.,2.,3.6]])
    sum1 = 0
    for i in range(m):
        sum2 = 0
        for j in range(4):
            sum2 += (X[j] - C[j][i])**2
        sum1 += 1./(sum2 + beta[i])
    return -sum1

def shekel10(X):
    '''http://www.sfu.ca/~ssurjano/shekel.html'''
    m = 10
    beta = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    C = a([[4.,1.,8.,6.,3.,2.,5.,8.,6.,7.],
           [4.,1.,8.,6.,7.,9.,5.,1.,2.,3.6],
           [4.,1.,8.,6.,3.,2.,3.,8.,6.,7.],
           [4.,1.,8.,6.,7.,9.,3.,1.,2.,3.6]])
    sum1 = 0
    for i in range(m):
        sum2 = 0
        for j in range(4):
            sum2 += (X[j] - C[j][i])**2
        sum1 += 1./(sum2 + beta[i])
    return -sum1

def hartman6(X):
    '''http://www.sfu.ca/~ssurjano/hart6.html'''
    alpha = (1.0, 1.2, 3.0, 3.2)
    A = a([[10, 3, 17, 3.5, 1.7, 8],
           [0.05, 10, 17, 0.1, 8, 14],
           [3.0, 3.5, 1.7, 10, 17, 8],
           [17, 8, 0.05, 10, 0.1, 14]])
    P = a([[0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
           [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
           [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
           [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]])
    sum1 = 0
    for i in range(4):
        sum2 = 0
        for j in range(6):
            sum2 += A[i][j] * (X[j] - P[i][j])**2
        sum1 += alpha[i] * np.exp(-sum2)
    return -sum1

def jennrich_sampson(X):
    '''http://infinity77.net/global_optimization/test_functions_nd_J.html'''
    x1, x2 = X
    sum1 = 0
    for i in range(1, 11):
        sum1 += (2 + 2*i - (np.e**(i*x1) + np.e**(i*x2)))**2
    return sum1

def centered_jennrich_sampson(X):
    '''http://infinity77.net/global_optimization/test_functions_nd_J.html'''
    x1, x2 = X
    x1 += 0.5
    x2 += 0.5
    sum1 = 0
    for i in range(1, 11):
        sum1 += (2 + 2*i - (np.e**(i*x1) + np.e**(i*x2)))**2
    return sum1

##########  Gradients  ##########
def get_grad(f_name):
    grads = {
        'rastrigin': rastrigin_grad,
    }
    return grads[f_name]

def rastrigin_grad(X, A=10):
    x1, x2 = X
    g1 = 2*pi*A * sin(2*pi*x1) + 2*x1
    g2 = 2*pi*A * sin(2*pi*x2) + 2*x2
    return g1, g2

def find_L(f_name):
    D = get_D(f_name)
    grad = get_grad(f_name)
    lb = get_lb(f_name)
    ub = get_ub(f_name)

    def negative_grad_norm(X, grad, lb, ub):
        for i in range(len(lb)):
            if lb[i] > X[i] or ub[i] < X[i]:
                return float('inf')
        return -enorm(grad(X))
    Ls = []
    # Naudoti globalaus, o ne lokalaus optimizavimo algoritma
    Ls.append(minimize(negative_grad_norm, lb, options={'disp': False},
                       args=(grad, lb, ub)).fun)
    Ls.append(minimize(negative_grad_norm, (a(lb)+a(ub))/2., options={'disp': False},
                       args=(grad, lb, ub)).fun)
    Ls.append(minimize(negative_grad_norm, ub, options={'disp': False},
                       args=(grad, lb, ub)).fun)
    return -min(Ls)


########  Function parameters  ##########
functions = [
    # 'rastrigin': rastrigin,
    # 'hyperparabola': hyperparabola,
    # 'rosenbrock': rosenbrock,
    # 'styblinski': styblinski,
    ###### Straipsnio funkcijos ######
    ('branin', branin),
    ('goldstein_price', goldstein_price),
    ('six_hump_camel_back', six_hump_camel_back),
    ('shubert', shubert),
    ('alolyan', alolyan),
    ('easom', easom),
    ('rastrigin', rastrigin),
    ('hartman3', hartman3),
    ('shekel5', shekel5),
    ('shekel7', shekel7),
    ('shekel10', shekel10),
    ('hartman6', hartman6),
    ('reduced_shekel5', shekel5),
    ('reduced_shekel7', shekel7),
    ('reduced_shekel10', shekel10),
    ('jennrich_sampson', jennrich_sampson),
    ('centered_jennrich_sampson', centered_jennrich_sampson),
]

def get_D(f_name):
    Ds = {
        'hartman3': 3,
        'shekel5': 4,
        'shekel7': 4,
        'shekel10': 4,
        'hartman6': 6,
        'reduced_shekel5': 4,
        'reduced_shekel7': 4,
        'reduced_shekel10': 4,
    }
    if not Ds.has_key(f_name):
        return 2
    return Ds[f_name]

def get_lb(f_name, D=2):
    lbs = {
        'hyperparabola': [-2.]*D,
        'rosenbrock': [-1.]*D,
        'styblinski': [-5]*D,
        'branin': [-5, 0],
        'goldstein_price': [-2., -2.],
        'six_hump_camel_back': [-3., -2.],
        'shubert': [-10.]*D,
        'alolyan': [-1.]*D,
        'easom': [-30.]*D,
        'rastrigin': [-5.]*D,
        'hartman3': [0.]*D,
        'shekel5': [0.]*D,
        'shekel7': [0.]*D,
        'shekel10': [0.]*D,
        'hartman6': [0.]*D,
        'jennrich_sampson': [0.]*D,
        'centered_jennrich_sampson': [-0.5]*D,
        'reduced_shekel5': [0]*D,
        'reduced_shekel7': [0]*D,
        'reduced_shekel10': [0]*D,
    }
    return lbs[f_name][:D]

def get_ub(f_name, D=2):
    ups = {
        'hyperparabola': [2.]*D,
        'rosenbrock': [1.]*D,
        'styblinski': [5]*D,
        'branin': [10, 15],
        'goldstein_price': [2., 2.],
        'six_hump_camel_back': [3., 2.],
        'shubert': [10]*D,
        'alolyan': [1.]*D,
        'easom': [30.]*D,
        'rastrigin': [5.]*D,
        'hartman3': [1.]*D,
        'shekel5': [10.]*D,
        'shekel7': [10.]*D,
        'shekel10': [10.]*D,
        'hartman6': [1.]*D,
        'jennrich_sampson': [1.]*D,
        'centered_jennrich_sampson': [0.5]*D,
        'reduced_shekel5': [4.]*D,
        'reduced_shekel7': [4.]*D,
        'reduced_shekel10': [4.]*D,
    }
    return ups[f_name][:D]

def get_min(f_name, D=2):
    minimums = {
        'rastrigin': [[0]*(D+1)],
        'hyperparabola': [[1]*D + [0]],
        'rosenbrock': [[1]*D + [0]],
        'branin': [-np.pi, 12.275, 0.397887],
        'goldstein_price': [0., -1., 3.],
        'six_hump_camel_back': [0.0898, -0.7126, -1.0316],
        'shubert': [4.85805,5.4828, -186.7309],
        'alolyan': [-1/3., 1., -1.18519],
        'easom': [np.pi, np.pi, -1.000],
        'rastrigin': [0, 0, 0],
        'hartman3': [0.114614, 0.555649, 0.852547, -3.86278],
        'shekel5':  [4., 4., 4., 4., -10.1532],
        'shekel7':  [4., 4., 4., 4., -10.4029],
        'shekel10': [4., 4., 4., 4., -10.5364],
        'hartman6': [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573, -3.322],
        'jennrich_sampson': [0.257825, 0.257825, 124.3621824],
        'centered_jennrich_sampson': [-0.242175, -0.242175, 124.3621824],
        'reduced_shekel5': [4., 4., 4., 4., -10.1532],
        'reduced_shekel7': [4., 4., 4., 4., -10.4029],
        'reduced_shekel10': [4., 4., 4., 4., -10.5364],
    }
    return minimums[f_name]

def get_L(f_name, C=1):
    '''Lipschitz constant:   L = max_{x in D} ||grad(f)(x)||'''
    Ls = {
        14.1421356237
#         'hyperparabola': [4]*C,  # L = 2*ub[0]
#         'rosenbrock': [500]*C,     # L = 100*x2 - 600*x1^2*x2 + 500*x1^4 + 2*x1 + 2
#         'styblinski': [1]*C,     # L = 3*x - 32*x + 5
#         'shubert': [1]*C,     # L = ??
    }
#     if not Ls.has_key(f_name):
#         return None
#     if C == 1:
#         return Ls[f_name][0]
    return Ls[f_name][:C]

# def get_error(f_name):
#     errors = {
#         'rastrigin': 10**0,
#         'hyperparabola': 2 * 10**-1,
#         'rosenbrock': 3 * 10**1,
#         'styblinski': 10**-1,
#         'shubert': 10**-1, # ??
#     }
#     return errors[f_name]


if __name__ == '__main__':
    print enorm(rastrigin_grad([4.25, 4.25]))
    print find_L('rastrigin')
    exit()

    C = 1
    max_f_calls = 1000
    stats = []

    f_name = 'reduced_shekel10'
    f = dict(functions)[f_name]
    D = get_D(f_name)
    lb = get_lb(f_name, D)
    ub = get_ub(f_name, D)
    f_min = get_min(f_name)
    print f_min, f(f_min[:-1])
    print 'Minimum OK:', round(f(f_min[:-1]), 4) == round(f_min[-1], 4)
    # draw_3d_objective_function(f, lb, ub)
    exit()

    # Paleisti eksperimentus su sukalibruotomis funkcijomis.
    # Todo: sukalibruoti 'rosenbrock' ir 'styblinski' funkcijas.

    for f_name, f in functions.items():
        lb = get_lb(f_name, D)
        ub = get_ub(f_name, D)
        # L = get_L(f_name, C)
        # error = get_error(f_name)
        f_min = get_min(f_name)[-1]
        error = 1.0
        # print f_name, lb, ub, L, error, fmin
        # draw_3d_objective_function(f, lb, ub, title=f_name)
        start = datetime.now()
        pareto_front, simplexes, f_calls = disimpl_2v(f, lb, ub, error, max_f_calls, f_min)
        end = datetime.now()
        # show_partitioning(simplexes)
        stats.append((f_name, f_calls))
        print '%s: f_calls: %s, duration: %s' % (f_name, f_calls, end-start)
