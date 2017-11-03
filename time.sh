# for N in 8 128 1024 2048; do
#     for a in fft fft_gpu dft_gpu dft; do
#         { time ./fft -a$a -t5 -p -N$N; } | python3 stats.py
#     done
# done

file=stats-tesla

echo algorithm,n,mean,stdev,max > $file

for N in 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 ; do
    for a in fft fft_gpu; do
        { time ./fft -a$a -t10 -p -N$(($N)); } | python3 stats.py $file $a $N
    done
done

for N in 256 512 1024 2048 4096; do
    for a in dft_gpu dft; do
        { time ./fft -a$a -t10 -p -N$(($N)); } | python3 stats.py $file $a $N
    done
done

