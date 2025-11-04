beta = 1 / 0.6 # 1 / kT
Lmax = 30
em_Mu, en_Mp, en_Mc = 0.0, 0.0, 0.0  # Example energy values
import numpy as np

def B(x):
    return np.exp(- beta * x)

def f_2(i, j, h, l, OMM):
    return 1.0 # Placeholder for the actual function
   
def fill_barPs(n, d, xi, P, OMM, M, bar_xi, bar_P, bar_Pm, bar_Pm1, bar_M, em_Mu, en_Mp, en_Mc, Lmax):
    # 対角線 d の全てを埋める: bar_P, bar_Pm, bar_Pm1
    return 

def fill_barM(n, d, xi, P, OMM, M, bar_xi, bar_P, bar_M, em_Mu, en_Mp, en_Mc, Lmax):
    # 対角線 d の全てを埋める
    return 

def outside(n, xi, P, OMM, M, en_Mu, en_Mp, en_Mc, Lmax):
    # init ...
    bar_xi = [0.0]*(n+1); bar_xi[n] = 1.0
    bar_P  = np.zeros((n+1, n+1))
    bar_M = np.zeros((3, n+1, n+1))  # bar_M[0]: M0, bar_M[1]: M1, bar_M[2]: M2

    # 集約テーブルの初期化
    bar_Pm  = np.zeros((n+1, n+1))
    bar_Pm1 = np.zeros((n+1, n+1))

    # (A) xi recursion
    for i in range(1, n + 1): # i = 1, 2, ..., n (0-origin)
        bar_xi[i] += xi[i-1] + sum([
            bar_xi[j] * P[j, i-1] for j in range(0, i-1) #j = 0, 1, ..., i - 2
        ]) # summation はベクトルを作って合計取る

    # (B) span-descending pass for bar_P and M_bar
    for d in range(n - 1, 0, -1): # jax.lax.scan などを使う
       # 計算グラフが大きくなりすぎるため、勾配チェックポインティングを使う
       fill_barPs(n, d, xi, P, OMM, M, bar_xi, bar_P, bar_Pm, bar_Pm1, bar_M, em_Mu, en_Mp, en_Mc, Lmax)
       fill_barM(n, d, xi, P, OMM, M, bar_xi, bar_P, bar_M, em_Mu, en_Mp, en_Mc, Lmax)
       
 
    Z = xi[n]
    post = [[0.0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            post[i][j] = (P[i][j] * bar_P[i][j]) / Z
    return post, bar_xi, bar_P, bar_M