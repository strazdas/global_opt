import os
import numpy as np
dirs = ['common__v_v2__md_nmd__e1_e0.01',
 'gkls_cls1__v_v2__md_nmd__1_0.01',
 'gkls_cls2__v_v2__md_ndm__1_0.01',
 'gkls_cls3__v_v2__md_nmd__1_0.01',
 'gkls_cls4__v_v2__md_nmd__1_0.01',
 'gkls_cls5__v_v2__md_nmd__1_0.01']


# Get filename list
for cls in dirs:
    e1_stats = {
        'v2-md': {'best': 0, 'avg': 0},
        'v2-nmd': {'best': 0, 'avg': 0},
        'v-md': {'best': 0, 'avg': 0},
        'v-nmd': {'best': 0, 'avg': 0},
    }
    # e001_stats = {
    #     'v2-md': {'best': 0, 'avg': 0},
    #     'v2-nmd': {'best': 0, 'avg': 0},
    #     'v-md': {'best': 0, 'avg': 0},
    #     'v-nmd': {'best': 0, 'avg': 0},
    # }
    parsed = []
    for filename in os.listdir(cls):
        algorithm = None
        if '_False_' in filename and '-2v_' in filename:
            algorithm = 'v2-nmd'
        elif '_True_' in filename and '-2v_' in filename:
            algorithm = 'v2-md'
        elif '_True_' in filename and '-v_' in filename:
            algorithm = 'v-md'
        elif '_False_' in filename and '-v_' in filename:
            algorithm = 'v-nmd'
        if 'common' in cls:
            f_name = filename.split('_')[-3]
            evaluations = int(filename.split('_')[-1])
        else:
            f_name = filename.split('_')[2]
            evaluations = int(filename.split('_')[-1])

        if evaluations > 10000:
            evaluations = 10000
        if 'e0.01' in filename:
        # if 'e1.00' in filename:
            parsed.append((algorithm, f_name, evaluations))

    for algorithm in ['v-nmd', 'v-md', 'v2-nmd', 'v2-md']:
        for f_name in list(set([p[1] for p in parsed])):
            evals = {'v-nmd': 0, 'v-md': 0, 'v2-nmd': 0, 'v2-md': 0}
            for a, f, e in parsed:
                if f == f_name:
                    evals[a] = e
            if algorithm == min(evals, key=lambda x: evals[x]):
                if evals[algorithm] < 10000:
                    e1_stats[algorithm]['best'] += 1
        f_calls = []
        for a, f, e in parsed:
            if a == algorithm:
                f_calls.append(e)
        e1_stats[algorithm]['avg'] = np.mean(f_calls)


    print cls.split('__')[0],
    for a in ['v2-nmd', 'v2-md', 'v-nmd', 'v-md']:
        # print '%s  ' % ( str(e1_stats[a]['best'],)),
        print '%s  ' % ( str(e1_stats[a]['avg'],)),
    print

    # import ipdb; ipdb.set_trace()

            # If best algorithm then best += 1
