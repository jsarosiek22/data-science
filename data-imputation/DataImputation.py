import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
df = pd.read_csv('/home/incomplete.csv')
dfc = pd.read_csv('/home/complete.csv')
r = df.shape[0]
c = df.shape[1]
print(f'#s Row: {r}, # Columns: {c}')
print(f'# Missing Entries: {df.isna().sum().sum()}')
print(f'% Missing Entries: {df.isna().sum().sum()/(r*c)}%')
print(f'# Missing Entries by Column:\n{df.isna().sum()}')
print(f'% Missing Entries by Column:\n{df.isna().sum()/r}')


def extend_df(df):
    import pandas as pd
    import numpy as np
    r = df.shape[0]
    c = df.shape[1]
    df_out = pd.DataFrame(np.nan, index=range(r+2), columns=range(c+2))
    for i in range(r):
        for j in range(c):
            df_out.iloc[i+1, j+1] = df.iloc[i, j]
    return(df_out)


toy_df = pd.DataFrame([[None, None, 1], [None, None, 2], [3, 4, 5]])
toy_extended = extend_df(toy_df)
display(toy_df)
display(toy_extended)


def neighbor_mean(df_extended, i, j):
    import pandas as pd
    if df_extended.isna().iloc[i+1, j+1] == 0:
        return None
    else:
        block = df_extended.iloc[i:(i+3), j:(j+3)]
        n_miss = block.isna().sum().sum()
        if n_miss == 9:
            return None
        else:
            return(block.sum().sum()/(9-n_miss))


display(toy_df)
print(neighbor_mean(toy_extended, 0, 0))
print(neighbor_mean(toy_extended, 1, 0))
print(neighbor_mean(toy_extended, 0, 1))
print(neighbor_mean(toy_extended, 1, 1))


def neighbor_impute(df):
    import numpy as np
    import pandas as pd
    df_out = df.copy(deep=df.isna().sum().sum())
    df_extended = extend_df(df)
    na_ind = np.where(df.isnull())
    n_miss = len(na_ind[0])
    for k in range(n_miss):
        i = na_ind[0][k]
        j = na_ind[1][k]
        df_out.iloc[i, j] = neighbor_mean(df_extended, i, j)
    print(f'# Unimputed entries: {i*j}')
    return(df_out)


print(neighbor_impute(toy_df))

imputed_neighbor = neighbor_impute(df)

i = 0
while imputed_neighbor.isna().sum().sum() > 0:
    i += 1
    imputed_neighbor = neighbor_impute(imputed_neighbor)
    print(f'Iteration #{i}')

df_filled = df.fillna(0, inplace=False)


plt.imshow(df_filled, cmap='gray')


def Rmean_impute(df):
    import numpy as np
    df_copy = df.copy(deep=True)
    Rmean = df_copy.mean(axis=1)
    r = df_copy.shape[0]

    na_ind = np.where(df_copy.isnull())

    t = len(na_ind[0])

    for k in range(t):
        i = na_ind[0][k]
        j = na_ind[1][k]
        df_copy.iloc[i, j] = Rmean[i]

    return(df_copy)


imputed_Rmean = Rmean_impute(df)

plt.imshow(imputed_Rmean, cmap='gray')

plt.imshow(imputed_neighbor, cmap='gray')

plt.imshow(dfc, cmap='gray')

n_miss = df.isna().sum().sum()
sq_diff_Rmean = (dfc - imputed_Rmean)**2
sq_diff_neighbor = (dfc - imputed_neighbor)**2
MSE_Rmean = sq_diff_Rmean.sum().sum()/n_miss
MSE_neighbor = sq_diff_neighbor.sum().sum()/n_miss
RMSE_Rmean = MSE_Rmean ** 0.5
RMSE_neighbor = MSE_neighbor ** 0.5
print(f'RMSE of imputation by row means: {round(RMSE_Rmean,2)}')
print(f'RMSE of imputation by neighbors: {round(RMSE_neighbor,2)}')
