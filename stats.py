import sys
from statistics import mean, median, stdev

data = [float(d.split()[0]) for d in sys.stdin.readlines()]

print("{:.5} +- {:.5},   {:.5} to {:.5},   median: {:.5},   sum: {:.5}".format(
    mean(data), stdev(data), min(data), max(data), median(data), sum(data)))

if len(sys.argv) == 4:
    with open(sys.argv[1], 'a+') as f:
        f.write(', '.join([str(e) for e in [sys.argv[2], sys.argv[3], mean(data), stdev(data), max(data)]]) + '\n')
