
```math
\begin{align}
\text{scaled 外側アルゴリズム with } &  x_i, x_j \text{ terms:} \\
\bar{\xi}'(i) & = s(1) \bar \xi' (i - 1) + 
\sum_{\substack{j \\ j < i - 1}} 
\sum_{\text{bp_idx}_{j, i - 1} ( = (b_j, b_{i-1}))}
\bar \xi'(j) P'(\text{bp_idx}_{j, i-1}, j, i - 1) x_{j, b_j} x_{i, b_{i-1}}\\

\bar P'(\text{bp_idx}_{h, l}, h, l) & := \bar \xi'(h) \xi'(l + 1) 
+ \sum_{\substack{i, j \\ i < h, l < j \\ h - i - 1 + j - l - 1 \leq 30}}  
\sum_{\text{bp_idx}_{i, j} ( = (b_i, b_j))} B(f_2(i, j, h, l)) \bar P'(\text{bp_idx}_{i, j}, i, j) s(h - i + j - l)  x_{i, b_i}  x_{j, b_j}\\

& \quad + \sum_{\substack{i \\ i < h}} \left[
\begin{array}[l]
 & s(1)M'(1, i + 1, h - 1) \bar P'_{m1} (i,l)  \\
 + \left\{ s(1) M'(1, i + 1, h - 1)
 + s(h - i)B(M_u(h - i - 1)) \right\} \bar P'_{m} (i, l) 
\end{array} 
\right] \\

\bar P'_m (i, l) & := \sum_{\substack{j \\ l < j}} s(1) M'(1, l+1, j-1) 
\sum_{\text{bp_idx}_{i, j} ( = (b_i, b_j))} 
B(M_p(b_i, b_j)) \bar P'(\text{bp_idx_ij}, i, j) x_{i, b_i} x_{j, b_j} \\

\bar P'_{m1} (i, l) & := \sum_{\substack{j \\ l < j}} s(j - l) B(M_u (j - l - 1))
\sum_{\text{bp_idx}_{i, j} ( = (b_i, b_j))}
B(M_p(b_i, b_j)) \bar P'(\text{bp_idx_ij}, i, j) x_{i, b_i} x_{j, b_j} \\

\bar M'(2, h, l) & := s(1) \bar M'(2, h-1, l) B(M_u) 
+ s(2) \sum_{\text{bp_idx}_{h-1, l+1} ( = (b_{h-1}, b_{l+1}))} 
    \bar P(h - 1, l + 1) B(M_c(b_{h-1}, b_{l+1})) 
    x_{h-1, b_{h-1}} x_{l + 1, b_{l+1}}
    \\

\bar M'(1, h, l) & := s(1) \bar M'(1, h-1, l) B(M_u) 
+ \sum_{i < h-1}  \bar M'(2, i, l)
\sum_{\text{bp_idx}_{i, h-1} ( = (b_{i}, b_{h-1}))} 
P'(\text{bp_idx}_{i, h-1}, i, h-1) B(M_p(b_i, b_{h-1})) x_{i, b_i} x_{h-1, b_{h-1}}  \\

\bar M'(0, h, l) & := s(1) \bar M'(0, h-1, l) B(M_u) 
+ \sum_{i < h-1} \left( \bar M'(0, i, l) + \bar M'(1, i, l) \right)
\sum_{\text{bp_idx}_{i, h-1} ( = (b_{i}, b_{h-1}))} 
P'(\text{bp_idx}_{i, h-1}, i, h-1) B(M_p(b_i, b_{h-1}))  x_{i, b_i} x_{h-1, b_{h-1}} 
\end{align}
```



