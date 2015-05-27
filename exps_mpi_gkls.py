#! /home/albertas/how_to_use/env/bin/python
from mpi4py import MPI
from sys import argv

from experiments import *
from disimpl_2v import disimpl_2v
from disimpl_v import disimpl_v


algorithms = {
    'disimpl-v': disimpl_v,
    'disimpl-2v': disimpl_2v,
}


if __name__ == '__main__':
    from experiments import functions, get_D, get_lb, get_ub, get_min, gkls_function
    from utils import wrap

    max_f_calls = 15000
    algorithm = 'disimpl-v'
    error = 1.0
    mirror_division = False

    if len(argv) == 2:
        error = argv[1]
    elif len(argv) == 3:
        error = float(argv[1])
        mirror_division = bool(argv[2])
    elif len(argv) == 4:
        error = float(argv[1])
        if argv[2] == 'False':
            mirror_division = False
        else:
            mirror_division = True
        algorithm = argv[3]

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    gkls_cls = 1
    gkls_fid_from = 1 + rank * 10
    gkls_fid_till = 1 + (rank+1) * 10

    for gkls_cls in [5]:
        for gkls_fid in range(gkls_fid_from, gkls_fid_till):
            f_name = 'gkls_cls%d_%d' % (gkls_cls, gkls_fid)
            D = get_D(f_name)
            lb = get_lb(f_name, D)
            ub = get_ub(f_name, D)
            f = wrap(gkls_function, lb, ub, gkls_cls=gkls_cls, gkls_fid=gkls_fid)   # f = wrap(dict(functions)[f_name], lb, ub)

            min_x = get_min(f_name, D)[:-1]
            min_f = get_min(f_name, D)[-1]
            # draw_3d_objective_function(branin, lb, ub)
            start = datetime.now()
            pareto_front, simplexes, f = algorithms[algorithm](f, lb, ub, error, max_f_calls, min_f, mirror_division)
            end = datetime.now()

            output_path = '/home/albertas/global_opt/results/cls%d/%s__e%.2f_%s__%s__%d' % (gkls_cls, f_name, error, algorithm, str(mirror_division), f.calls)
            output = open(output_path, 'w+')
            output.write('%s  %d  %f  %s\n' % (f_name, f.calls, f.min_f, str(f.min_x)))
            output.write('duration: %s\n' % (end-start,))
            output.write('Actual minimum: %f, %s' % (min_f, str(min_x)))
            output.write('D: %d,  lb: %s,  ub: %s' % (D, lb, ub))
            output.close()
