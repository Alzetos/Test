import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter


class Solution:
    def __init__(self):
        self.df = pd.read_csv("task1.csv", encoding="utf-8")
        self.df_active = None

    def create_fig(self, axisX, axisY, LABEL, COLOR, title, xlabel, ylabel):
        fig = plt.scatter(self.df[axisX], self.df[axisY], label=LABEL, color=COLOR)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.show()

    def rolling_median(self, window=5):
        self.df["y_median"] = self.df["y"].rolling(window=window, center=True).median()
        self.df["y_median"] = self.df["y_median"].fillna(
            self.df["y"]
        )  # fill borders via y_vals
        return self.df

    def derivative(self, window_length=99, polyorder=3):
        self.df["dy/dt"] = savgol_filter(
            self.df["y_median"],
            window_length=window_length,
            polyorder=polyorder,
            deriv=1,
            delta=(self.df["t"].iloc[1] - self.df["t"].iloc[0]),
        )  # savgol_filter for smoothing dy/dt after rolling, we need this for creating anamorph ln(1/y * dy/dt) ~ t
        return self.df

    def create_anamorph_1(self, t_start, t_end):
        mask = (self.df["t"] >= t_start) & (self.df["t"] <= t_end)
        self.df_active = self.df[mask].copy()
        self.df_active["Y_anamorph"] = np.log(
            self.df_active["dy/dt"] / self.df_active["y_median"]
        )
        return self.df_active

    def find_anamorph1(self):
        df_active = self.df_active.dropna()

        X = df_active[["t"]]
        y = df_active["Y_anamorph"]

        model = LinearRegression()
        model.fit(X, y)

        r_squared = model.score(X, y)
        print(f"determination coef R^2 = {r_squared:.4f}")

        slope = model.coef_[0]
        print(f"Slope k_1 = {-slope}")

        A = model.intercept_
        print(f"A = {A}")
        print(f"equation: ln((dy/dt) / y) = {(model.intercept_)} - {-slope}*t")

        print("_" * 50)

        y_pred = model.predict(X)

        plt.scatter(X, y, color="blue", label="Data")
        plt.plot(X, y_pred, color="red", linewidth=2, label="Regression")
        plt.legend()
        plt.show()

    def find_k_y_2(self):
        df_active = self.df_active.dropna()

        X = np.log(df_active[["y_median"]])
        y = df_active["dy/dt"] / df_active["y_median"]

        model = LinearRegression()
        model.fit(X, y)

        r_squared = model.score(X, y)
        print(f"determination coef R^2 = {r_squared:.4f}")
        if r_squared < 0.9:
            print("narrow time borders")
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
        plt.legend()
        plt.show()

    def find_k_y_3(self):
        # Y = mX + b => в y_inf X=Y=ln(y_inf)
        # X = ln y(t)
        # Y = ln y(t + tau), if tau = 3
        # m = exp(-k*tau)

        # time shift
        tau = 3
        self.df_active["y(t+tau)"] = self.df_active["y_median"].shift(-tau)
        df_active = self.df_active.dropna()

        X = np.log(df_active[["y_median"]])  # get y(t)
        y = np.log(df_active["y(t+tau)"])

        model = LinearRegression()
        model.fit(X, y)

        print(f"R^2 = {model.score(X, y)}")

        m = model.coef_[0]

        k = (np.log(m)) / (-tau * (df_active["t"].iloc[1] - df_active["t"].iloc[0]))
        print(f"Slope k_3 = {k}")

        intercept = model.intercept_
        b = intercept
        ln_y_inf = (b) / (1 - m)
        y_inf = np.exp(ln_y_inf)
        print(f"y_inf = {y_inf}")

        y_pred = model.predict(X)

        plt.scatter(X, y, color="blue", label="Data")
        plt.plot(X, y_pred, color="red", linewidth=2, label="Regression")
        plt.legend()
        plt.show()

        print("_" * 50)

    def check(self):
        y_inf_1 = 10525.319190026761
        y_inf_2 = 10137.722710843114

        self.df_active["z_1"] = np.log(np.log(y_inf_1 / self.df_active["y_median"]))

        X = self.df_active[["t"]]
        y = self.df_active["z_1"]

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
        plt.legend()
        plt.show()

        self.df_active["z_2"] = np.log(np.log(y_inf_2 / self.df_active["y_median"]))

        X_2 = self.df_active[["t"]]
        y_2 = self.df_active["z_2"]

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
        plt.show()

        equation_1 = b - k * self.df_active["t"].iloc[25]
        equation_2 = b_2 - k_2 * self.df_active["t"].iloc[25]
        Y_1 = self.df_active["z_1"].iloc[25]
        Y_2 = self.df_active["z_2"].iloc[25]

        # СХООООООООООООООООДИТСЯ УРАААААААААААААААААААА

        print(f"{Y_1} = {equation_1}")
        print(f"{Y_2} = {equation_2}")


if __name__ == "__main__":
    try:
        sol = Solution()
    except FileNotFoundError:
        pass

    print("EDA")
    sol.rolling_median(window=5)

    sol.derivative(window_length=99, polyorder=3)

    print("Active area")
    sol.create_anamorph_1(t_start=9, t_end=16)

    print("Params")

    print("\n1 method")
    sol.find_anamorph1()

    print("\n2 method")
    sol.find_k_y_2()

    print("\n3 method")
    sol.find_k_y_3()

    print("\ncheck")
    sol.check()
