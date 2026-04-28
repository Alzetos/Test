import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter

df = pd.read_csv("task1.csv", encoding="utf-8")

fig = plt.plot(df["t"], df["y"], label="", color="b")
plt.title("Figure y(t) - original")
plt.xlabel("Axis t")
plt.ylabel("Axis y")
plt.grid(True)
plt.show()

df["y_median"] = df["y"].rolling(window=5, center=True).median()  # rolling median

df["y_median"] = df["y_median"].fillna(df["y"])  # fill borders via y_vals

fig = plt.plot(df["t"], df["y_median"], label="", color="b")
plt.title("Figure y_median(t)")
plt.xlabel("Axis t")
plt.ylabel("Axis y")
plt.grid(True)
plt.show()

df["dy/dt"] = savgol_filter(
    df["y_median"],
    window_length=99,
    polyorder=3,
    deriv=1,
    delta=(df["t"].iloc[1] - df["t"].iloc[0]),
)  # savgol_filter for smoothing dy/dt after rolling, we need this for creating anamorph ln(1/y * dy/dt) ~ t

fig = plt.plot(df["t"], df["dy/dt"], label="dy/dt", color="b")
plt.title("Figure dy/dt от t")
plt.xlabel("Axis t")
plt.ylabel("Axis dy/dt")
plt.grid(True)
plt.show()

df_active = df.copy()
df_active["Y_anamorph"] = np.log(df_active["dy/dt"] / df_active["y_median"])

fig = plt.plot(df_active["t"], df_active["Y_anamorph"], label="dy/dt", color="b")
plt.title("Figure Y(t)")
plt.xlabel("Axis t")
plt.ylabel("Axis Y_anamorph")
plt.grid(True)
plt.show()

mask = (df["t"] >= 9) & (df["t"] <= 16)  # interval of uniformity => "+"
df_active = df_active[mask].copy()
fig = plt.plot(df_active["t"], df_active["Y_anamorph"], label="dy/dt", color="r")
plt.title("Uniformity figure Y(t) (t in [9;16])")
plt.xlabel("Axis t")
plt.ylabel("Axis Y(t)")
plt.grid(True)
plt.show()


# 1 now we have to find "Y = ln(A) - kt". k calculate as slope tg(angle) with using linear regression
def find_k_1(df_active):
    model = LinearRegression()
    df_active = df_active.dropna()

    X = df_active[["t"]]
    y = df_active["Y_anamorph"]

    model.fit(X, y)
    r_squared = model.score(X, y)
    print(f"determination coef R^2 = {r_squared:.4f}")
    sclope = model.coef_[0]
    print(f"Sclope k_1 = {-sclope}")

    A = model.intercept_  # intercept = ln(A)
    print(f"equation ln((dy/dt) / y) = {A} - {-sclope}*t")
    print("_" * 50)
    y_pred = model.predict(X)

    plt.scatter(X, y, color="blue", label="Data")
    plt.plot(X, y_pred, color="red", linewidth=2, label="Regression")
    plt.xlabel("t")
    plt.ylabel("ln((1/y) * (dy/dt))")
    plt.legend()
    plt.show()


find_k_1(df_active)


# 2 k and y_inf via (3.14)
def find_k_y_2(df_active):
    df_active = df_active.dropna()

    X = np.log(df_active[["y_median"]])
    y = df_active["dy/dt"] / df_active["y_median"]

    model = LinearRegression()
    model.fit(X, y)

    r_squared = model.score(X, y)
    print(f"determination coef R^2 = {r_squared:.4f}")
    sclope = model.coef_[0]
    print(f"Slope k_2 = {-sclope}")

    k = -sclope
    intercept = model.intercept_  # k*ln(y_inf)
    ln_y_inf = intercept / k  # ln(y_inf)
    y_inf = np.exp(ln_y_inf)  # y_inf
    print(f"y_inf = {y_inf}")
    print("_" * 50)

    y_pred = model.predict(X)

    plt.scatter(X, y, color="blue", label="Data")
    plt.plot(X, y_pred, color="red", linewidth=2, label="Regression")
    plt.xlabel("ln(y_median(t))")
    plt.ylabel("(1/y) * (dy/dt)")
    plt.legend()
    plt.show()


find_k_y_2(df_active)


def find_k_y_3(df_active):
    # Y = mX + b => в y_inf X=Y=ln(y_inf)
    # X = ln y(t)
    # Y = ln y(t + tau), if tau = 3
    # m = exp(-k*tau)
    tau = 3
    df_active["y(t+tau)"] = df_active["y_median"].shift(-tau)  # time shift
    df_active = df_active.dropna()
    X = np.log(df_active[["y_median"]])  # get y(t)
    y = np.log(df_active["y(t+tau)"])

    model = LinearRegression()
    model.fit(X, y)
    print(f"R^2 = {model.score(X, y)}")

    m = model.coef_[0]
    k = (np.log(m)) / (-tau * (df_active["t"].iloc[1] - df_active["t"].iloc[0]))
    print(f"k3 = {k}")

    intercept = model.intercept_
    b = intercept
    ln_y_inf = (b) / (1 - m)
    y_inf = np.exp(ln_y_inf)
    print(f"y_inf = {y_inf}")

    print("_" * 50)

    y_pred = model.predict(X)

    plt.scatter(X, y, color="blue", label="Data")
    plt.plot(X, y_pred, color="red", linewidth=2, label="Regression")
    plt.xlabel("ln(y_median(t))")
    plt.ylabel("ln(y(t+tau))")
    plt.legend()
    plt.show()


find_k_y_3(df_active)


def check(df_active):
    y_inf_1 = 10525.319190026761
    y_inf_2 = 10137.722710843114

    df_active["z_1"] = np.log(np.log(y_inf_1 / df_active["y_median"]))

    X = df_active[["t"]]
    y = df_active["z_1"]

    model = LinearRegression()  # Y=b-kX
    model.fit(X, y)

    print(f"R^2 = {model.score(X, y)}")
    k = -model.coef_[0]
    b = model.intercept_

    print(f"k_eval_1 = {k}")
    print(f"b_eval_1 = {b}")

    print(f"equation Y = {b} - {k}*t")

    y_pred1 = model.predict(X)

    plt.scatter(X, y, color="blue", label="Data")
    plt.plot(X, y_pred1, color="red", linewidth=2, label="Regression")
    plt.xlabel("t")
    plt.ylabel("z_1")
    plt.legend()
    plt.show()

    df_active["z_2"] = np.log(np.log(y_inf_2 / df_active["y_median"]))

    X_2 = df_active[["t"]]
    y_2 = df_active["z_2"]

    model = LinearRegression()  # Y=b-kX
    model.fit(X_2, y_2)

    print(f"R^2 = {model.score(X_2, y_2)}")
    k_2 = -model.coef_[0]
    b_2 = model.intercept_

    print(f"k_eval_2 = {k_2}")
    print(f"b_eval_2 = {b_2}")

    print(f"equation Y = {b_2} - {k_2}*t")

    y_pred_2 = model.predict(X_2)

    plt.scatter(X_2, y_2, color="blue", label="Data")
    plt.plot(X_2, y_pred_2, color="red", linewidth=2, label="Regression")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("z_2")
    plt.show()

    equation_1 = b - k * df_active["t"].iloc[25]
    equation_2 = b_2 - k_2 * df_active["t"].iloc[25]
    Y_1 = df_active["z_1"].iloc[25]
    Y_2 = df_active["z_2"].iloc[25]

    print(f"{Y_1} = {equation_1}")
    print(f"{Y_2} = {equation_2}")


check(df_active)
