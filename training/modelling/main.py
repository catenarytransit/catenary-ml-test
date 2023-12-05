import train_algos
import reading_data

"""
V_max (velocity clamping): 5 for small (1-4), 15 for medium (5-8), 25 for large (>8)
"""

if __name__ == '__main__':
    U_train, Y_train, U_val, Y_val, U_test, Y_test = reading_data.read_data()
    L, U, w = train_algos.pso_algorithm(U_train.to_numpy(), Y_train.to_numpy(), U_val.to_numpy(), Y_val.to_numpy(),
                              s_min=4, s_max=50, P=20, c_1=0.05, c_2=0.05, V_max=5,
                              max_simulation_step=8000, R_norm_threshold=0.1)
    print(L, U, w)

