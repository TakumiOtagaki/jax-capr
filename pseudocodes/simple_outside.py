def outside(n, xi, P, M0, M1, M2, B_Mu, B_Mp, B_Mc,
            B_f2, Lmax):
    """
    xi[t]      : prefix partition up to t (xi[0]=1, xi[n]=Z)
    P[i,j]     : Z^b(i,j)
    M0/1/2[i,j]: M(p,i,j) for p=0,1,2
    B_f2(i,j,h,l): Boltzmann weight for two-loop (i,j)-(h,l) excl. OMM if separated
    Lmax       : internal loop size limit (e.g., 30); None if not used
    """
    # init
    xi_bar = [0.0]*(n+1); xi_bar[n] = 1.0
    P_bar  = [[0.0]*(n+1) for _ in range(n+1)]
    M2_bar = [[0.0]*(n+1) for _ in range(n+1)]
    M1_bar = [[0.0]*(n+1) for _ in range(n+1)]
    M0_bar = [[0.0]*(n+1) for _ in range(n+1)]

    # 2) barP と集約テーブル（全ゼロで開始）
    barPm  = [[0.0]*n for _ in range(n)]   # \bar P^m(i,l)
    barPm1 = [[0.0]*n for _ in range(n)]   # \bar P^{m1}(i,l)

    # (A) external loop outside (right -> left)
    for r in range(n-1, -1, -1):  # r = i in text
        # unpaired propagate
        xi_bar[r] += xi_bar[r+1]
        # paired cases: (i,k) ends at r<k<=n
        for k in range(r+1, n+1):
            P_bar[r][k] += xi_bar[k+1] * xi[r-1]  # ext-left * ext-right
            xi_bar[r-1] += xi_bar[k+1] * P[r][k]

    # (B) span-descending pass for P_bar and M_bar
    for d in range(n, 1, -1):              # span length
        # --- precompute barP^m and barP^{m1} for this diagonal, as functions of (i,l) ---
        # barPm[i][l] = sum_{j>l} P_bar[i][j] * M1[l+1][j-1]
        # barPm1[i][l]= sum_{j>l} P_bar[i][j] * (B_Mu)^(j-l-1)

        # --------- for LLM, fill here -------
        # ------------------------------------

        for i in range(1, n-d+2):
            j = i + d - 1
            # (B1) P -> child M2
            if i+1 <= j-1:
                M2_bar[i+1][j-1] += P_bar[i][j] * (B_Mc * B_Mp)

            # (B2) P -> child P (two-loop)
            for h in range(i+1, j-1):
                for l in range(h+1, j):
                    if Lmax is None or (h-i-1)+(j-l-1) <= Lmax:
                        P_bar[h][l] += P_bar[i][j] * B_f2(i,j,h,l)

            # (B3) P -> child P (multiloop via barPm, barPm1)
            for h in range(i+1, j):
                l = h  # placeholder; we add contributions for all (h,l) later
            # Actually apply:
            for h in range(i+1, j-1):
                for l in range(h+1, j):
                    add = (B_Mc * B_Mp) * (
                        M1[i+1][h-1] * barPm1[i][l]
                        + (M1[i+1][h-1] + (B_Mu ** (h-i-1))) * barPm[i][l]
                    )
                    P_bar[h][l] += add

    # (C) M_bar propagation (left boundary descending)
    for p, Mbar in [(2, M2_bar), (1, M1_bar), (0, M0_bar)]:
        for l in range(1, n+1):
            for h in range(l, 0, -1):    # h descending
                # unpaired in multiloop
                if h-1 >= 1:
                    Mbar[h-1][l] += Mbar[h][l] * B_Mu
                # paired branch P(i,k) inside M
                for i in range(1, h):
                    k = h-1
                    if i < k:
                        P_bar[i][k] += Mbar[h][l] * B_Mp * (M1 if p==2 else M0)[k+1][l]

    Z = xi[n]
    # posterior
    post = [[0.0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            post[i][j] = (P[i][j] * P_bar[i][j]) / Z
    return post, xi_bar, P_bar, (M0_bar, M1_bar, M2_bar)