import csv
import numpy as np
import matplotlib.pyplot as plt

with open('stats-peano2X') as f:
    data = dict()
    ns = set() 
    for row in csv.DictReader(f):
        if not row['algorithm'] in data:
            data[row['algorithm']] = []
        data[row['algorithm']].append({key.lstrip().rstrip(): float(v) for key, v in row.items() if key != 'algorithm'})
        ns.add(int(row['n']))

"""
for k in ['fft']:
    # TODO how to measure speedup?
    speedup = [(r['n'], r['mean'] / rg['mean'])
            for r, rg in
            zip(sorted(data[k],          key=lambda v: v['n']),
                sorted(data[k + '_gpu'], key=lambda v: v['n']),)]
    plt.scatter(*zip(*speedup))
    plt.title(k)
    plt.show()
"""

xs = np.array([r['n'] for r in data['fft']])
ys1 = np.array([r['mean'] for r in data['fft']])
ys2 = np.array([r['mean'] for r in data['fft_gpu']])
speedup = ys1 / ys2

# 1024 threads after n >= 1024
p = np.array([1024 if x >= 1024 else x for x in xs])
efficiency = speedup / p

plt.scatter(xs, speedup, alpha=0.5, linewidth=2)
plt.ylabel("Speedup")
plt.xlabel("N")
plt.title("Speedup = $S = T_s / T_p$")
plt.show()

plt.scatter(xs, efficiency, alpha=0.5, linewidth=2)
plt.ylabel("Efficiency")
plt.xlabel("N")
plt.title("Efficiency = $S / p$, p = min(N, 1024)")
plt.show()

# table of runtimes
d = sorted(list(data.items()), reverse=True)
print(d)
print('| N | ' + (' | '.join([k for k, _ in d])), '|')
print('-' * 79)
for n in sorted(list(ns)):
    print('| {:5}'.format(n), end=' | ')
    for v in [v for _,vs in d for v in vs if n == int(v['n']) ]:
        print('{: 7.4} \\pm {:2.2}'.format(v['mean'], v['stdev']), end=' | ')
    print()

