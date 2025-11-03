beta = 1 / 0.6 # 1 / kT
Lmax = 30
import numpy as np

def B(x):
    return np.exp(- beta * x)

def f_2(i, j, h, l, OMM):
    return 1.0 # Placeholder for the actual function

def outside(n, xi, P, OMM, M, en_Mu, en_Mp, en_Mc, Lmax):
    # init ...
    bar_xi = [0.0]*(n+1); bar_xi[n] = 1.0
    bar_P  = np.zeros((n+1, n+1))
    bar_M = np.zeros((3, n+1, n+1))  # bar_M[0]: M0, bar_M[1]: M1, bar_M[2]: M2

    # 集約テーブルの初期化
    bar_Pm  = np.zeros((n+1, n+1))   # Σ_{j>l} bar_P[i][j] * M1[l+1][j-1]
    bar_Pm1 = np.zeros((n+1, n+1))   # Σ_{j>l} bar_P[i][j] * B_Mu**(j-l-1)

    # (A) xi recursion
    for i in range(1, n + 1): # i = 1, 2, ..., n (0-origin)
        bar_xi[i] += xi[i-1] + sum([
            bar_xi[j] * P[j, i-1] for j in range(0, i-1) #j = 0, 1, ..., i - 2
        ])

    # (B) span-descending pass for bar_P and M_bar
    for d in range(n - 1, 0, -1):
        for h in range(1, n - d + 1):
            l = h + d
            # ------- P --------
            # bar_P(h, l)
            bar_P[h, l] += bar_xi[h] * xi[l+1] \
                + sum([
                    B(f_2(i, j, h, l, OMM)) * bar_P[i, j]
                    for i in range(1, h) for j in range(l + 1, n + 1)
                ])
            # bar_P^m(h, l)
            bar_Pm[h, l] += sum([
                bar_P[h, j] * M[1, l + 1, j - 1]
                for j in range (l + 1, n + 1)
            ])
            # bar_P^{m+1}(h, l)
            bar_Pm1[h, l] += sum([
                bar_P[h, j] * (B(en_Mu) ** (j - l - 1))
                for j in range (l + 1, n + 1)
            ])
            # ------- M0, M1, M2 -------
            # bar_M2(h, l)
            bar_M[2, h, l] += bar_M[2, h - 1, l] * B(en_Mu) + bar_P[h - 1, l + 1] * B(en_Mc + en_Mp)
            # bar_M1(h, l)
            bar_M[1, h, l] += bar_M[1, h - 1, l] * B(en_Mu) + sum([
                P[i, h - 1] * B(en_Mp) * bar_M[2, i, l]
                for i in range(1, h - 1)
            ])
            # bar_M0(h, l)
            bar_M[0, h, l] += bar_M[0, h - 1, l] * B(en_Mu) + sum([
                P[i, h - 1] * B(en_Mp) * ( bar_M[1, i, l] + bar_M[0, i, l])
                for i in range(1, h - 1)
            ])

    Z = xi[n]
    post = [[0.0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            post[i][j] = (P[i][j] * bar_P[i][j]) / Z
    return post, bar_xi, bar_P, bar_M