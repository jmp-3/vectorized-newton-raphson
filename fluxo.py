import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=5, suppress=True, linewidth=120)
print("\n====== comeco ======\n")


G = np.zeros((3, 3))

B = np.array([[-12.5,   4.16,   8.33],
              [  4.16, -24.16, 20.0 ],
              [  8.33,  20.0, -28.33]])
print("B =\n", B)

Y = G + 1j * B
print("Y =\n", Y)


Sb = 50   # MVA
Vb = 120  # kV
#                    1        2        3
V     = np.array([1.0,     np.nan,  np.nan])
theta = np.array([0.0,     np.nan,  np.nan])
P     = np.array([np.nan,  1.5,    -1.0])
Q     = np.array([np.nan,  0.057,  -1.0])

v_2 = 1.0
v_3 = 1.0
theta_2 = 0.0
theta_3 = 0.0

Pd = np.array([1.0, 0.0, 1.0])   # carga ativa por barra (pu)
Qd = np.array([0.5, 0.0, 1.0])   # carga reativa por barra (pu)
plot_j = []
plot_p = []
plot_q = []


for i in range(1, 31):

    print(f"\n====== iteracao numero: {i}======")
    V = np.array([1.0, v_2, v_3])
    theta = np.array([0.0, theta_2, theta_3])

    # ======= A partir daqui eh generico ==========
    Vkl = V[None, :] * V[:, None]          # gerar matrizes que combinam as tensoes
                                           # multiplicando as tensoes correspondentes
                                           # ao subindice da matriz (V.*V')

    n = len(V)
    Vkk = Vkl * np.eye(n)                  # apenas os quadrados na diagonal

    Vkl = Vkl * ~np.eye(n, dtype=bool)     # para o calculo das potencias nao se
                                           # considera os elementos da diagonal

    theta_kl = theta[None, :] - theta[:, None]   # elemento (k,m) = theta_m - theta_k
                                                 # (theta - theta')

    sin_theta_kl = np.sin(theta_kl)        # gerar matriz dos senos
    cos_theta_kl = np.cos(theta_kl)        # gerar matriz dos cosenos

    # Calculo dos delta p  (sum do MATLAB = soma das linhas = axis=0)
    delta_p = P - np.sum(Vkl * (G * cos_theta_kl + B * sin_theta_kl) + Vkk * G, axis=0)
    delta_q = Q - np.sum(Vkl * (G * sin_theta_kl - B * cos_theta_kl) - Vkk * B, axis=0)
    print("delta_p =", delta_p)
    print("delta_q =", delta_q)

    # Criar matrizes e variaveis uteis
    Xkk = np.eye(n)
    Xkm = (~np.eye(n, dtype=bool)).astype(float)   # fora da diagonal (k ~= m)

    Vkl = V[None, :] * V[:, None]          # Para o calculo dos valores na jacobiana
                                           # se consideram os da diagonal

    sin_km = sin_theta_kl.T                # elemento (k,m) = sin(theta_k - theta_m)
    cos_km = cos_theta_kl.T                # elemento (k,m) = cos(theta_k - theta_m)

    Vr = V[None, :]                        # V  (vetor-linha): elemento (k,m) = V_m
    Vc = V[:, None]                        # V' (vetor-coluna): elemento (k,m) = V_k

    # Calculo da jacobiana
    # dP/dTheta
    Hkk = (-B * Vkk - np.sum(Vkl * (G * sin_theta_kl - B * cos_theta_kl), axis=0)) * Xkk
    Hkm = Vkl * (G * sin_km - B * cos_km) * Xkm
    H = Hkm + Hkk

    # dP/dV
    Nkk = ((G * Vr) + np.sum(Vc * (G * cos_theta_kl + B * sin_theta_kl), axis=0)) * Xkk
    Nkm = Vc * (G * cos_km + B * sin_km) * Xkm
    N = Nkm + Nkk

    # dQ/dTheta
    Mkk = (-G * Vkk + np.sum(Vkl * (G * cos_theta_kl + B * sin_theta_kl), axis=0)) * Xkk
    Mkm = -Vkl * (G * cos_km + B * sin_km) * Xkm
    M = Mkm + Mkk

    # dQ/dV
    Lkk = (-B * Vr + np.sum(Vc * (G * sin_theta_kl - B * cos_theta_kl), axis=0)) * Xkk
    Lkm = Vc * (G * sin_km - B * cos_km) * Xkm
    L = Lkm + Lkk

    # ======= A partir daqui nao eh mais generico ==========
    # (indices em Python comecam em 0: barra 2 -> 1, barra 3 -> 2)

    jac = np.array([[H[1, 1], H[1, 2], N[1, 1], N[1, 2]],
                    [H[2, 1], H[2, 2], N[2, 1], N[2, 2]],
                    [M[1, 1], M[1, 2], L[1, 1], L[1, 2]],
                    [M[2, 1], M[2, 2], L[2, 1], L[2, 2]]])
    print("jac =\n", jac)

    plot_j.append(jac.flatten())           # vetores apenas para plotagem (linha a linha)
    plot_p.append(delta_p)
    plot_q.append(delta_q)

    # parar quando convergir
    if np.max(np.abs(np.r_[delta_p[1:3], delta_q[1:3]])) < 1e-8:
        break

    inv_jac = -np.linalg.pinv(jac)         # apenas para visualizacao da
                                           # inversa, se desejado

    F = -np.array([delta_p[1], delta_p[2], delta_q[1], delta_q[2]])
    X = np.array([theta_2, theta_3, v_2, v_3])

    X = X - np.linalg.pinv(jac) @ F        # em NumPy nao eh preciso transpor

    theta_2 = X[0]                         # atualizar
    theta_3 = X[1]
    theta_2_deg = X[0] * 180 / np.pi       # visualizacao apenas
    theta_3_deg = X[1] * 180 / np.pi       # visualizacao apenas
    v_2 = X[2]
    v_3 = X[3]
    print(f"theta_2_deg = {theta_2_deg:.4f}")
    print(f"theta_3_deg = {theta_3_deg:.4f}")
    print(f"v_2 = {v_2:.4f}")
    print(f"v_3 = {v_3:.4f}")


# plotar figuras de convergencias
plot_j = np.array(plot_j)
plot_p = np.array(plot_p)
plot_q = np.array(plot_q)
it = np.arange(1, len(plot_p) + 1)

plt.figure(1)
plt.plot(it, plot_p, linewidth=2)
plt.legend(["delta P1", "delta P2", "delta P3"], loc="center right")
plt.title("Convergência Delta P")
plt.xlabel("n iterações")

plt.figure(3)
plt.plot(it, plot_q, linewidth=2)
plt.legend(["delta Q1", "delta Q2", "delta Q3"], loc="center right")
plt.title("Convergência Delta Q")
plt.xlabel("n iterações")

plt.figure(2)
keys = ["H22", "H23", "N22", "N23", "H32", "H33", "N32", "N33",
        "M22", "M23", "L22", "L23", "M32", "M33", "L32", "L33"]
plt.plot(it, plot_j, linewidth=2)
plt.legend(keys, loc="center right")
plt.title("Parametros da matriz jacobiana")
plt.xlabel("n iterações")


# calculo das potencias
V = np.array([1.0, v_2, v_3])              # usar os valores da ultima atualizacao
theta = np.array([0.0, theta_2, theta_3])

n = len(V)
Vkl = V[None, :] * V[:, None]
Vkk = Vkl * np.eye(n)
Vkl = Vkl * ~np.eye(n, dtype=bool)
theta_kl = theta[None, :] - theta[:, None]
sin_theta_kl = np.sin(theta_kl)
cos_theta_kl = np.cos(theta_kl)

print("\npu")
Pcalc = np.sum(Vkl * (G * cos_theta_kl + B * sin_theta_kl) + Vkk * G, axis=0)
Qcalc = np.sum(Vkl * (G * sin_theta_kl - B * cos_theta_kl) - Vkk * B, axis=0)
Pg = Pcalc + Pd                            # potencia gerada = injecao liquida + carga
Qg = Qcalc + Qd
print("Pcalc =", Pcalc)
print("Qcalc =", Qcalc)
print("V =", V)
print("Pg =", Pg)
print("Qg =", Qg)

print("\nMVA e kV")
print("Pcalc =", Pcalc * Sb)
print("Qcalc =", Qcalc * Sb)
print("V =", V * Vb)
print("Pg =", Pg * Sb)
print("Qg =", Qg * Sb)

print("\n====== fim ======\n")

plt.show()
