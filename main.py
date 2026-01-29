import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import os
from mpl_toolkits.mplot3d import Axes3D
import argparse

FIGURES_DIR = "figures"
SHOW_PLOTS = True  # True to show plots, False to just save them
# Adjust parameters as needed in methods simulation() (toy problem) and cases() (EDF cases)

def save_figure(name, fig=None, show=False, pad_inches=0):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.pdf"), bbox_inches="tight", pad_inches=pad_inches)
    print(f"Figure enregistrée : {os.path.join(FIGURES_DIR, f'{name}.pdf')}")
    if show:
        plt.show()
    else:
        plt.close(fig)


# Rtot value
def Rtot(e, lamb, Rp):
    Rtot = e / lamb + Rp
    return Rtot


# dTdx value
def dTdx(T, e, lamb_f, Rp, r, m, Cp, Text):
    Rt = Rtot(e, lamb_f, Rp)
    dTdx = (2 * math.pi * r) / (m * Cp * Rt) * (Text - T)
    return dTdx


# dedt value
def dedt(T, e, lamb_f, E, R, k25, e_inf):
    dedt = k(T, E, R, k25) / lamb_f * (e_inf - e) * e
    return dedt


# k value
def k(T, E, R, k25):
    k = k25 * math.exp(-E/R * (1/T - 1/298.15))
    return k


# Simulates T and e in function of time
# Returns the linear space of time and x
# and the matrices T and x
def simulate_T_e(T0, e0, Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max = 3.1, t_max = 60, density = 1000):
    dx = x_max/density
    x = np.arange(0, x_max, dx)

    dt = t_max/density
    t = np.arange(0, t_max, dt)

    # T: time x position
    T = [[T0 for _ in range(len(x))] for _ in range(len(t))]
    e = [[e0 for _ in range(len(x))] for _ in range(len(t))]

    for i in range(1, len(t)):
        for j in range(1, len(x)):
            T[i][j] = (T[i][j-1] + dTdx(T[i][j-1], e[i][j-1], lamb_f, Rp, r, m, Cp, Text) * dx)
            e[i][j] = (e[i-1][j] + dedt(T[i-1][j], e[i-1][j], lamb_f, E, R, k25, e_inf) * dt)

    T = np.array(T)
    e = np.array(e) * pow(10, 6)

    return T, e, x, t


# Plots e in function of x for fixed values of time
def plot_e_x(e, t, x):
    thickness = 2

    percentages = [0/50, 5/50, 10/50, 15/50, 20/50, 25/50, 30/50, 35/50, 40/50, 49.9/50]
    days = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50]

    for i in range(len(percentages)):
        plt.plot(x[1:], e[int(len(t)*percentages[i]), 1:], label=f'{days[i]} jours', linewidth=thickness)
    plt.ylabel(r'$e\,(\mu \text{m})$')
    plt.xlabel('x (m)')
    plt.legend(loc='center left', fontsize="8")
    plt.grid()
    save_figure('plot_e_x', show=SHOW_PLOTS)


# Plots e in function of time for fixed percentages of total lenght of the pipe
def plot_e_t(e, t, x):
    thickness = 2

    percentages = [1e-3, 0.25, 0.5, 0.75, 0.99]
    percent = [1e-3, 25, 50, 75, 100]

    for i in range(len(percentages)):
        plt.plot(t, e[:, int(len(x) * percentages[i])], label=f'{percent[i]}%', linewidth=thickness)
    plt.ylabel(r'$e\,(\mu \text{m})$')
    plt.xlabel('t (jours)')
    plt.legend(loc='center left', fontsize="8")
    plt.grid()
    save_figure('plot_e_t', show=SHOW_PLOTS)


# Plots e in function of x for fixed values of time
def plot_T_x(T, t, x):
    thickness = 2

    percentages = [0/50, 5/50, 10/50, 15/50, 20/50, 25/50, 30/50, 35/50, 40/50, 49.9/50]
    days = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50]

    for i in range(len(percentages)):
        plt.plot(x[1:], T[int(len(t)*percentages[i]), 1:] - 273.15, label=f'{days[i]} jours', linewidth=thickness)
    plt.ylabel(r'$\text{T}\,(^o\text{C})$')
    plt.xlabel('x (m)')
    plt.legend(loc='center left', fontsize="8")
    plt.grid()
    save_figure('plot_T_x', show=SHOW_PLOTS)


# Plots e in function of time for fixed percentages of total lenght of the pipe
def plot_T_t(T, t, x):
    thickness = 2
    
    percentages = [1e-3, 0.25, 0.5, 0.75, 0.99]
    percent = [1e-3, 25, 50, 75, 100]

    for i in range(len(percentages)):
        plt.plot(t[1:], T[1:, int(len(x) * percentages[i])] - 273.15, label=f'{percent[i]}%', linewidth=thickness)
    plt.ylabel(r'$\text{T}\,(^o\text{C})$')
    plt.xlabel('t (jours)')
    plt.legend(loc='center left', fontsize="8")
    plt.grid()
    save_figure('plot_T_t', show=SHOW_PLOTS)


# Plots both eXt and eXx
def plot_eXt_eXx(T0, e0, Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max = 3.1, t_max = 60, density = 1000):
    _, e, x, t = simulate_T_e(T0, e0, Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)

    plot_e_t(e, t, x)
    plot_e_x(e, t, x)


# Plots both TXt and TXx
def plot_TXt_TXx(T0, e0, Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max = 3.1, t_max = 60, density = 1000):
    T, _, x, t = simulate_T_e(T0, e0, Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)

    plot_T_t(T, t, x)
    plot_T_x(T, t, x)

# solution analytique
def MA(x,Th,Te,r,m,cp,Rp):
    return Th+(Te-Th)*np.exp(-(2 * np.pi * r *x/ (m * cp * Rp)))
def T(Nx,Nt,L,lambda_f,r,m,cp,Th,Rp,Te,e0,dt,E,R,k25,e_inf):
    T = np.zeros((Nt, Nx))
    ef = np.zeros((Nt, Nx))
    T[0, :] = Te
    ef[0, :] = e0
    dx=L/Nx

    for n in range(Nt):
        T[n, 0] = Te
        for i in range(Nx - 1):
            Rtot = Rp + ef[n, i] / lambda_f
            T[n, i + 1] = (
                T[n, i] - dx * (2 * np.pi * r / (m * cp * Rtot)) * (T[n, i] - Th))
        if n==Nt-1: break
        for i in range(Nx):
            ef[n + 1, i] = (
                ef[n, i]+ dt * (k(T[n, i],E,R,k25) / lambda_f)* (e_inf - ef[n, i]) * ef[n, i])
    return T 

def plot_T_different_dx(T,t,pas,L,Th,Te,r,m,cp,Rp,Nt,lambda_f,e0,dt,E,R,k25,e_inf):
    """
    Température en fonction de x pour différents pas dx 
    """
    plt.figure()
    for dx in pas:
        Nx=int(L/dx)
        x = np.linspace(0, L, Nx)
        it = int(0 * (len(t) - 1))
        plt.plot(x, T(Nx,Nt,L,lambda_f,r,m,cp,Th,Rp,Te,e0,dt,E,R,k25,e_inf)[it, :] - 273.15, label= f"dx = {dx} m")
    x=x = np.linspace(0, L, 1000)
    plt.plot(x,MA(x,Th,Te,r,m,cp,Rp)-273.15,label="modele analytique")
    plt.xlabel("x (m)")
    plt.ylabel("Température (°C)")
    plt.legend()
    plt.grid()
    save_figure('plot_T_different_dx', show=SHOW_PLOTS)

# Give different plots of e in function of x in the last instant of time for different
# numbers of points to use in simulation
def plot_sensibility(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max = 3.1, t_max = 60):
    thickness = 2

    densities = [50, 100, 150, 300, 500, 750, 1000]
    for density in densities:
        T, _, x, t = simulate_T_e(T0, e0, Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)
        t_plot = t
        x, t = np.meshgrid(x, t)
        plt.plot(t_plot[1:], T[1:, -1] - 273.15, label=f'{density} points', linewidth=thickness)

    plt.ylabel(r'$T_f\,(^o\text{C})$')
    plt.xlabel('t (jours)')
    plt.legend(loc='best', fontsize="8")
    plt.grid()
    save_figure('plot_sensibility', show=SHOW_PLOTS)


# Plots de experimental data in addition to our model varying the
# parameters k25 and E
def plot_nebot(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf):
# Nebot data (hardcoded)
    time_nebot = [0.190481, 0.761924, 1.33337, 1.71433, 2.2873, 3.05379, 3.82334, 4.76812, 5.91863, 7.06304, 7.82648, 8.97089, 9.93244, 11.0769, 12.0323, 13.3687, 14.1474, 14.9078, 15.877, 17.3963, 17.9799, 19.1197, 20.0767, 20.0797, 21.0428, 21.9967, 23.1335, 23.9016, 25.0444, 26.1843, 26.9477, 28.0921, 29.0522, 30.2011, 31.1505, 31.917, 34.2043, 34.983, 35.9262, 37.2688, 38.0413, 38.9831, 39.9507, 40.9108, 42.3965, 43.352, 44.1215, 44.6868, 46.4073, 47.1585, 47.7589, 49.0816, 50.2367, 51.1876, 52.1354, 52.8852, 53.8467, 54.9911, 56.1157, 56.8853, 57.9961, 58.9333]

    R_nebot = [1.26987e-05, 5.07949e-05, 8.88911e-05, 0.000114289, 0.00415249, 0.0162036, 0.0362549, 0.0163179, 0.0363946, 0.0404709, 0.0445218, 0.0485981, 0.0726622, 0.0767385, 0.0848022, 0.0928912, 0.136943, 0.132994, 0.177058, 0.16516, 0.197199, 0.189275, 0.201338, 0.209339, 0.237403, 0.241466, 0.225542, 0.241593, 0.24167, 0.233746, 0.237797, 0.241873, 0.261937, 0.278013, 0.270077, 0.282128, 0.28628, 0.330332, 0.306395, 0.330485, 0.358536, 0.330599, 0.370663, 0.390727, 0.290826, 0.29889, 0.318941, 0.302979, 0.319094, 0.291144, 0.367184, 0.339272, 0.371349, 0.367413, 0.355476, 0.323526, 0.34759, 0.351666, 0.303741, 0.323792, 0.239866, 0.199929]

    # Nebot plot
    thickness = 2
    plt.plot(time_nebot, R_nebot, 's')

    # k25 tests
    list_k25 = [700, 1100, 1500]
    for current_k25 in list_k25:
        k25 = current_k25
        _, e, x, t  = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)
        Rb = e/(lamb_f * 1000)
        plt.plot(t, Rb[:, int(len(x)/2)], label=fr'$k25 = {k25}\,\mathrm{{W/m^2/K/s}}$', linewidth=thickness)

    plt.ylabel(r'$R_f\,(m^2 K/kW)$')
    plt.xlabel('t (jours)')
    plt.legend(loc='center left', fontsize="8")
    plt.grid()
    save_figure('plot_nebot_k25', show=SHOW_PLOTS)

    k25 = 1100
    plt.plot(time_nebot, R_nebot, 's')

    # E tests
    list_E = [30000, 40000, 50000]
    for current_E in list_E:
        E = current_E
        _, e, x, t  = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)
        Rb = e/(lamb_f * 1000)
        plt.plot(t, Rb[:, int(len(x)/2)], label=fr'$E = {E}\,\mathrm{{J/mol}}$', linewidth=thickness)

    plt.ylabel(r'$R_f\,(m^2 K/kW)$')
    plt.xlabel('t (jours)')
    plt.legend(loc='center left', fontsize="8")
    plt.grid()
    save_figure('plot_nebot_E', show=SHOW_PLOTS)

    E = 40000
    plt.figure()
    plt.plot(time_nebot, R_nebot, 's')

    # e0 tests
    list_e0 = [5e-7, 2e-6, 8e-6]
    for current_e0 in list_e0:
        e0 = current_e0
        _, e, x, t = simulate_T_e(T0, e0, Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)
        Rb = e / (lamb_f * 1000)
        plt.plot(t, Rb[:, int(len(x)/2)], label=fr"$e_0 = {e0 * 1e6}\,\mathrm{{\mu m}}$", linewidth=thickness)

    plt.ylabel(r'$R_f\,(m^2 K/kW)$')
    plt.xlabel('t (jours)')
    plt.legend(loc='center left', fontsize="8")
    plt.grid()
    save_figure('plot_nebot_e0', show=SHOW_PLOTS)


# Do a 3D plot of the behavior of the temperature in the system
def plotT3D(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf):
    T, _, x, t = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)
    x, t = np.meshgrid(x, t)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(t, x, T - 273.15, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')

    ax.set_xlabel('t (jours)')
    ax.set_ylabel('x (m)')
    ax.set_zlabel(r'T ($^oC$)')

    save_figure('plotT3D', fig, show=SHOW_PLOTS, pad_inches=0.15)


# Do a 3D plot of the behavior of the biofilm thickness in the system
def plote3D(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf):
    _, e, x, t = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)
    x, t = np.meshgrid(x, t)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(t, x, e, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.view_init(elev=45, azim=-135)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')

    ax.set_xlabel('t (jours)')
    ax.set_ylabel('x (m)')
    ax.set_zlabel(r'$e\ (\mu m)$')

    save_figure('plote3D', fig, show=SHOW_PLOTS, pad_inches=0.15)

def plot_T_x_t(L, r, mdot, cp, Te, Text, R_propre, lambda_f, e_inf, e0, k25, E, R, dt=3600):    
    Nt=int(30*24*3600/dt)           
    t = np.arange(Nt) * dt

    def k_T(T):
        return k25 * np.exp(-E/R * (1/T - 1/298.15))
    def MA(x):
        return Text+(Te-Text)*np.exp(-(2 * np.pi * r *x/ (mdot * cp * R_propre)))

    def T(Nx,Nt):
        T = np.zeros((Nt, Nx))
        ef = np.zeros((Nt, Nx))
        T[0, :] = Te
        ef[0, :] = e0
        dx=L/Nx

        for n in range(Nt):
            T[n, 0] = Te
            for i in range(Nx - 1):
                Rtot = R_propre + ef[n, i] / lambda_f
                T[n, i + 1] = (
                    T[n, i] - dx * (2 * np.pi * r / (mdot * cp * Rtot)) * (T[n, i] - Text))
            if n==Nt-1: break
            for i in range(Nx):
                # Prevent overflow by limiting ef[n, i] and using a smaller dt if necessary
                growth = dt * (k_T(T[n, i]) / lambda_f) * (e_inf - ef[n, i]) * ef[n, i]
                # Clamp growth to avoid overflow
                if abs(growth) > 1e-3:
                    growth = np.sign(growth) * 1e-3
                ef[n + 1, i] = ef[n, i] + growth
                # Optionally, clamp ef[n + 1, i] to physical limits
                if ef[n + 1, i] < 0:
                    ef[n + 1, i] = 0
                if ef[n + 1, i] > e_inf:
                    ef[n + 1, i] = e_inf
        return T     

    def plot_T_vs_x_at_t(T,t,pas,inst):
        """
        Température en fonction de x pour différents instants t
        instants : liste de fractions du temps total
        """
        fig = plt.figure()
        for dx in pas:
            Nx=int(L/dx)
            x = np.linspace(0, L, Nx)
            it = int(inst * (len(t) - 1))
            plt.plot(x, T(Nx,Nt)[it, :] - 273.15, label= f"dx = {dx} m")
        x=x = np.linspace(0, L, 1000)
        plt.plot(x,MA(x)-273.15,label="modele analytique")
        plt.xlabel("x (m)")
        plt.ylabel("Température (°C)")
        plt.legend()
        plt.grid()
        save_figure('sensibilite_dx', fig, show=SHOW_PLOTS, pad_inches=0)

    instants_t = [0.0]
    pas=[2,1,0.5,0.1,0.05,0.01,0.005,0.001]
    # fractions du temps total
    plot_T_vs_x_at_t(T,t,pas, 0)

def power(N, m, Cp, Text, T):
    return N*m*Cp*(Text - T)


# Plots the temperature of the output of the pipes through time to different values of m
def plot_lastT_for_different_m(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density, water):
    thickness = 2

    m_list = [0.40, 0.45, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    for current_m in m_list:
        m = current_m
        T, _, _, t = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)
        plt.plot(t[1:], T[1:, -1] - 273.15, label=f'm = {m:.2f} kg/s', linewidth=thickness)
        Tf = T[-1, -1] - 273.15
        

    if water == 'sea':
        plt.axhline(27, linestyle='dashed') #Tmax=27
    elif water == 'river':
        plt.axhline(30, linestyle='dashed') #Tmax=30

    plt.ylabel(r'$T_s \> (°C)$')
    plt.xlabel('t (jours)')
    plt.legend(loc='best', fontsize="8")
    plt.grid()
    save_figure('plot_lastT_for_different_m', show=SHOW_PLOTS)


# Get the power of the case 0 to sea scenario
def get_power_case0_sea(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density,):
    T, _, _, t = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)
    t = t[1:]
    Tf = T[1:, -1] - 273.15
    p = []

    Ti = T0 - 273.15
    
    for i in range(len(t)):
        p.append(power(616, m, Cp, Tf[i], Ti) * pow(10, -6))

    return p, t


# Get the power of the case 1 to sea scenario
#calcul le cout annuel de biocide pour un débit donné Dm
def annual_biocide_total_cost(m_dot,N_tube,price_per_kg=1.0,dosage=5e-6,time_per_year = 30*3600*365):
    return m_dot * time_per_year * dosage * price_per_kg * N_tube

#Calcul la puissance d'un tube 
def tube_power_biocide(
    m,
    *,
    T0, Rp, lamb_f, r, Cp, Text, x_max,
    t_max_short=2, density=1000,
    e0=2e-6, E=40e3, R=8.314, k25=1100, e_inf=200e-6,
    plot=False,
    plot_profile=False,
    title_prefix=""
):
   
    
    T, e, _, t = simulate_T_e(
        T0, e0, Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf,
        x_max, t_max_short, density
    )

    # Température de sortie (dernier point spatial)
    Tf = T[-1, -1]
    Ti = T0

    # Puissance d'un tube (W)
    P_tube = m * Cp * (Tf - Ti)


    return P_tube, Tf, t, T

def plot_N_vs_m_biocide(*,scenario="sea", m_min=0.4, m_max=0.8, n_points=21,T0=290.15, Rp=0.0, lamb_f=0.6, r=0.01, Cp=4180.0, Text=320.15, x_max=10.0,
    t_max_short=365, density=1000,
    # paramètres biofilm / biocide
    e0=5e-6, E=40e3, R=8.314, k25=1100, e_inf=100e-6,
    # objectif
    P_target=20e6,
    filename="N_vs_m_biocide.png"
):
    """
    Calcule N(m)=ceil(P_target/P_tube(m)) avec le modèle biocide,
    trace N en fonction de m
    """

    # 1) grille de débits
    m_list = np.linspace(m_min, m_max, n_points)

    # 2) calcul N(m)
    N_list = []
    for m in m_list:
        P_tube, Tf, _, _ = tube_power_biocide(m,
            T0=T0, Rp=Rp, lamb_f=lamb_f, r=r, Cp=Cp, Text=Text, x_max=x_max,
            t_max_short=t_max_short, density=density,
            e0=e0, E=E, R=R, k25=k25, e_inf=e_inf,
            plot=False, plot_profile=False
        )

        if P_tube <= 0:
            N_list.append(np.nan)
        else:
            N_list.append(math.ceil(P_target / P_tube))

    N_list = np.array(N_list)

    # 3) choix du débit selon scénario
    scenario = scenario.lower().strip()
    if scenario == "sea":
        m_choice = m_max
    elif scenario == "river":
        m_choice = m_min
    else:
        raise ValueError('scenario doit être "sea" ou "river"')

    idx = np.argmin(np.abs(m_list - m_choice))
    N_choice = N_list[idx]

    # 4) plot N(m)
    plt.figure()
    plt.plot(m_list, N_list, marker="o")
    plt.scatter([m_list[idx]], [N_choice], s=80)
    plt.xlabel("Débit massique par tube m (kg/s)")
    plt.ylabel("Nombre de tubes requis N")
    plt.grid()
    name, _ = os.path.splitext(filename)
    save_figure(name, show=SHOW_PLOTS)

    # 5) résultat final uniquement
    print(f"Résultat final ({scenario}) :")
    print(f"  Débit retenu m = {m_list[idx]:.3f} kg/s")
    print(f"  Nombre de tubes requis N = {N_choice}")
    print(f"Température de sortie associée Tf = {Tf - 273.15:.2f} °C")


    return m_list[idx], N_choice

#Trace la puissance P(t) produite par UN tube traité au biocide
def plot_power_vs_time_biocide(*,m,T0, Rp, lamb_f, r, Cp, Text, x_max,
    t_max=360, density=1000,
    e0=2e-6, E=40e3, R=8.314, k25=1100, e_inf=100e-6,
    filename="Power_vs_time.pdf", save=True):

    T, e, x, t = simulate_T_e(
        T0, e0, Rp, lamb_f, r, m, Cp, Text,
        E, R, k25, e_inf,
        x_max=x_max, t_max=t_max, density=density
    )

    Tf_t = T[:, -1]                 # température de sortie
    P_t = m * Cp * (Tf_t - T0)      # puissance (W)

    plt.figure()
    plt.plot(t, P_t / 1e3)
    plt.xlabel("Temps (jours)")
    plt.ylabel("Puissance par tube (kW)")
    plt.grid()

    name, _ = os.path.splitext(filename)
    if save:
        save_figure(name, show=SHOW_PLOTS)

    return t, P_t

t, P_t = plot_power_vs_time_biocide(
    m=0.8,
    T0=12+273.15, Rp=5e-4, lamb_f=0.6, r=0.02, Cp=4184,
    Text=60+273.15, x_max=10,
    t_max=365,
    e0=2e-6, E=40e3, R=8.314, k25=1100, e_inf=100e-6,
    filename="Power_vs_time_biocide_sea.pdf", save=False)

t, P_t = plot_power_vs_time_biocide(
    m=0.4,
    T0=25+273.15, Rp=5e-4, lamb_f=0.6, r=0.02, Cp=4184,
    Text=60+273.15, x_max=10,
    t_max=365,
    e0=2e-6, E=40e3, R=8.314, k25=1100, e_inf=100e-6,
    filename="Power_vs_time_biocide_river.pdf", save=False)


# Get the power of the case 2 to sea scenario
def get_power_case2_sea(T0, e0,n,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density):
    T, _, _, t = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)
    t = t[1:]
    Tf = T[1:, -1] - 273.15

    p = []
    Ti = T0 - 273.15
    cleaning = False
    n_cleans = 0
    for i in range(len(t)):
        current_power = power(n, m, Cp, Tf[i], Ti) * pow(10, -6)
        if current_power <= 20 and not cleaning:
            cleaning = True
            Ti_cleaning = t[i]
            n_cleans += 1

        if cleaning == True:
            current_power = 20

        if cleaning: 
            if t[i] - Ti_cleaning >= 1:
                cleaning = False
                j = i
                while j != len(Tf):
                    Tf[j] = Tf[j-i]
                    j += 1

        p.append(current_power)

    
    return p, t,n_cleans


# Get the power of the case 2 to river scenario
def get_power_case2_river(T0, e0,n, Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density):
    T, _, _, t = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)
    t = t[1:]
    Tf = T[1:, -1] - 273.15

    p = []
    Ti = T0 - 273.15
    cleaning = False
    n_cleans = 0
    for i in range(len(t)):
        current_power = power(n, m, Cp, Tf[i], Ti) * pow(10, -6)
        if current_power <= 20 and not cleaning:
            cleaning = True
            Ti_cleaning = t[i]
            n_cleans += 1

        if cleaning == True:
            current_power = 20

        if cleaning: 
            if t[i] - Ti_cleaning >= 1:
                cleaning = False
                j = i
                while j != len(Tf):
                    Tf[j] = Tf[j-i]
                    j += 1

        p.append(current_power)

    
    return p, t, n_cleans


# Get the power of the case 3
def get_power_case3(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density, water):
    if water == 'mer' or water == 'sea':
        n_pipes = 699
        frequency = 2 * (density/365)
        n_groups = 12
    if water == 'riviere' or water == 'river':
        n_pipes = 876
        frequency = 2 * (density/365)
        n_groups = 12
    
    Tfs = []
    Ti_cleaning = []
    cleaning = []
    T, _, _, t = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)
    Tfs_reference = T[1:, -1] - 273.15
    for i in range(n_groups):
        T, _, _, t = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)
        Tfs.append(T[1:, -1] - 273.15)
        Ti_cleaning.append(0)
        cleaning.append(False)
    t = t[1:]

    p = []
    Ti = T0 - 273.15
    n_cleans = 0
    j = 1
    current_group_being_cleanned = 0
    for i in range(len(t)):

        current_power = 0
        for current_group_analyzed in range(len(Tfs)):
            if cleaning[current_group_analyzed] == False:
                current_power += power(int(n_pipes/n_groups), m, Cp, Tfs[current_group_analyzed][i], Ti) * pow(10, -6)

        if j >= frequency:
            j = 0
            Ti_cleaning[current_group_being_cleanned] = t[i]
            cleaning[current_group_being_cleanned] = True
            n_cleans += 1
            current_group_being_cleanned += 1
            if current_group_being_cleanned == n_groups: 
                current_group_being_cleanned = 0

        for current_group_analyzed in range(len(Tfs)):
            if cleaning[current_group_analyzed] == True and t[i] - Ti_cleaning[current_group_analyzed] >= 1:
                cleaning[current_group_analyzed] = False
                counter = i
                while counter != len(Tfs[current_group_analyzed]):
                    Tfs[current_group_analyzed][counter] = Tfs_reference[counter-i]
                    counter += 1

        p.append(current_power)
        j += 1

    print(f'Nonmbre de nettoyages: {n_cleans}')

    return p, t
# get the power of the case 4
def get_power_case4(n,T0,e0,Rp,lamb_f,r,m,Cp,Text,E,R,k25,e_inf,x_max,t_max,density):
    T,_,_,t=simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)
    t = t[1:]
    Tf = T[1:, -1] - 273.15
    p = []

    Ti = T0 - 273.15
    
    for i in range(len(t)):
        p.append(power(n, m, Cp, Tf[i], Ti) * pow(10, -6))

    return p, t
# Get the power of the case 0 to river scenario
def get_power_case0_river(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density):
    T, _, _, t = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)
    t = t[1:]
    Tf = T[1:, -1] - 273.15
    p = []

    Ti = T0 - 273.15
    
    for i in range(len(t)):
        p.append(power(850, m, Cp, Tf[i], Ti) * pow(10, -6))

    return p, t



# Plots the power over time
def plot_powerXt(p, t):
    thickness = 2

    plt.plot(t, p, linewidth=thickness)
    
    plt.ylabel(r'$Puissance \> (MW)$')
    plt.xlabel('t (jours)')
    plt.grid()
    save_figure('plot_powerXt', show=SHOW_PLOTS)


def cost_pump(n, v, m):
    rho = 1000
    mu = pow(10, -3)
    L = 10
    Dm = 0.02
    a = 130 * pow(10, -6) * 365 * 24 / 1000
    b = m/0.7
    c = n*L/Dm
    d = pow(v, 2) * rho / 2
    e_ = 0.3164 * pow((rho * Dm * v / mu), (-0.25))
    return (a*b*c*d*e_)
    

# Coût annuel de pompage (mdot = débit massique PAR TUBE)
def pumping_cost(
    mdot,          # [kg/s] PAR TUBE (pas total)
    N,             # nombre de tubes
    L,             # longueur d'un tube [m]
    D,             # diamètre intérieur [m]
    eta_pump=0.7,  # rendement pompe
    t_days=365,    # jours de fonctionnement par an
):
    # Constantes eau
    rho = 1000.0       # kg/m3
    mu = 1.0e-3        # Pa.s
    C_energy = 130e-6  # €/Wh

    # Débit volumique par tube
    Vdot_tube = mdot / rho  # m3/s

    # Section d'un tube
    A = math.pi * D**2 / 4

    # Vitesse moyenne dans un tube
    v = Vdot_tube / A

    # Débit volumique total (tous les tubes)
    Vdot_tot = Vdot_tube * N

    # facteur de friction
    Re = rho * D * v / mu
    f = 0.3164 * (Re ** (-0.25)) 

    # Perte de charge d'un tube
    dP = f * (L / D) * (rho * v**2 / 2)

    # Puissance électrique
    P_elec = (Vdot_tot * dP) / eta_pump  # W

    # Coût annuel
    C_pump = P_elec * (t_days * 24) * C_energy

    return C_pump
    

def simulation():
    
    # Parameters
    Rp = 5e-4                 # [m^2 * K / W]
    lamb_f = 0.6              # [W/m/K]
    r = 6e-3                  # [m]
    v = 1.85                  # [m/s]
    Ar = math.pi * pow(r, 2)  # [m^2]
    m = 1000 * Ar * v         # [kg/s]
    Cp = 4184                 # [J/kg/K]
    Text = 35 + 273.15        # [K]
    E = 40000                 # [J/mol]
    R = 8.314                 # [J/(mol * K)]
    k25 = 1100                # [m/s]
    e_inf = 200e-6            # [m]
    L=10                       #[m]
    # Initial conditions
    T0 = 17 + 273.15         # [K]
    e0 = 0.01 * e_inf        # [m]
    
    plotT3D(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)
    plote3D(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)
    plot_sensibility(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)
    plot_nebot(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)
    plot_eXt_eXx(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)
    plot_TXt_TXx(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)
    plot_T_x_t(L, r, m, Cp, T0, Text, Rp, lamb_f, e_inf, e0, k25, E, R)

    pas=[2,1,0.5,0.1,0.05,0.01,0.005,0.001]
    Nt = 1000
    dt = 30 / 1000
    t = np.arange(0, Nt * dt, dt)
  
    plot_T_different_dx(T, t, pas, L, Text, T0, r, m, Cp, Rp, Nt, lamb_f, e0, dt, E, R, k25, e_inf)


def cases():
#     # Constantes
    Dm = 0.02                    # [m]
    x_max = 10                   # [m]
    Th = 60 + 273.15             # [K]
    Cp = 4184                    # [J/kg/K]
    lamb = 0.6                   # [W/m/K]
    Rp = 5e-4                    # [m^2 * K / W]
    Ea = 40000                   # [J/mol]
    Rg = 8.314                   # [J/(mol * K)]
    k25 = 1100                   # [W/m^2/K/s]
    Ar = (math.pi * pow(Dm, 2)) /4
    e_inf = 200e-6
    t_max = 365
    e0 = 0.01*e_inf
    N=[550,600,800,1000,1200,1600,2000]
    m=0.8
    n = N[0]
    c=80e-6
    Rc=4e-4


    # Parameteres pour la mer
    M =[0.4,0.5,0.6,0.65,0.7,0.75,0.8]
    T0 = 12 + 273.15
   
    
    print('-------eau de mer----------')
#     Case 0
    plot_lastT_for_different_m(T0, e0, Rp, lamb, Dm/2, m, Cp, Th, Ea, Rg, k25, e_inf, x_max, t_max, 300, 'mer')
   
    p, t = get_power_case0_sea(T0, e0, Rp, lamb,Dm/2,m,Cp,Th,Ea,Rg,k25, e_inf, x_max, t_max, 1000)
    plot_powerXt(p, t)

#      Case 1
    t, P_t = plot_power_vs_time_biocide(
        m=0.8,
        T0=12+273.15, Rp=5e-4, lamb_f=0.6, r=0.02, Cp=4184,
        Text=60+273.15, x_max=10,
        t_max=365,
        e0=2e-6, E=40e3, R=8.314, k25=1100, e_inf=100e-6,
        filename="Power_vs_time_biocide_sea.pdf")
    
    m_choice, N_choice = plot_N_vs_m_biocide(
    scenario="sea",
    T0=12+273.15, Rp=5e-4, lamb_f=0.6, r=0.02, Cp=4184,
    Text=60+273.15, x_max=10,
    t_max_short=365,
    e0=5e-6, E=40e3, R=8.314, k25=1100, e_inf=100e-6,
    P_target=20e6,
    filename="N_vs_m_biocide_rivière.pdf"
)
    C = pumping_cost(
    mdot=0.8,   # kg/s PAR TUBE
    N=290,
    L=10.0,
    D=0.02,
    eta_pump=0.7,
    t_days=365
)

    print(f"Coût annuel de pompage (eau froide) : {C:.0f} € / an")
    
#      Cas 2
    
    p,t,n= get_power_case2_sea(T0, e0, n, Rp, lamb, Dm/2, m, Cp, Th,Ea, Rg, k25, e_inf, x_max, t_max, 1000)
    plot_powerXt(p,t)

    "Le code suivant sert a representer le nombre de nettoyage en fonction du debit et du nombre de tubes"
    Nc = np.zeros((len(M), len(N)))
    for i, m in enumerate(M):
        for j, n in enumerate(N):
            Nc[i, j] = get_power_case2_sea(T0, e0, n, Rp, lamb, Dm/2, m, Cp, Th,Ea, Rg, k25, e_inf, x_max, t_max, 1000)[2]  # Nc en MW
    N_grid, M_grid = np.meshgrid(N, M)
    plt.figure()
    plt.contourf(N_grid, M_grid, Nc, levels=20)
    plt.xlabel("Nombre de tubes n")
    plt.ylabel("Débit massique ṁ [kg/s]")
    plt.colorbar(label="M")
    save_figure('nombre_de_nettoyages', show=SHOW_PLOTS)
    

    C2 = pumping_cost(
    mdot=0.7,     # m3/s
    N=600,
    L=10.0,        # m
    D=0.02,        # m
    eta_pump=0.7,
    t_days=352)
    print(f"Coût annuel de pompage scenario 2 (mer) : {C2:.0f} € / an")

    # Cas 3
    p, t = get_power_case3(T0, e0,  Rp, lamb, Dm/2, m, Cp, Th, Ea, Rg, k25, e_inf, x_max, t_max, 3000, 'sea')
    plot_powerXt(p, t)
    # cas4
    P=[]
    for n in N:
        p,t=get_power_case4(n,T0,0,Rp+Rc,lamb,(Dm-c)/2,m,Cp,Th,Ea,Rg,k25,e_inf,x_max,t_max,3000)
        P.append(p[-1])
    plt.ylabel('Puissance en MW')
    plt.xlabel('nombre de tubes')
    plt.grid()
    plt.plot(N,P)
    save_figure('puissance_vs_nombre_de_tubes', show=SHOW_PLOTS)

    C4= pumping_cost(
    mdot=0.8,     # m3/s
    N=678,
    L=10.0,        # m
    D=0.02,        # m
    eta_pump=0.7,
    t_days=189)
    print(f"Coût annuel de pompage scenario 3 (mer) : {C4:.0f} € / an")

    #cas 4

    C6 = pumping_cost(
    mdot=0.4,     # m3/s
    N=650,
    L=10.0,        # m
    D=0.02,        # m
    eta_pump=0.7,
    t_days=365)
    print(f"Coût annuel de pompage scenario 4 (mer) : {C6:.0f} € / an")




    print('-------eau de rivier----------')
     # Parameters river
    m = 0.5
    T0 = 25 + 273.15
    v = m/(1000 * Ar)
    

#      Case 0
    plot_lastT_for_different_m(T0, e0, Rp, lamb, Dm/2, m, Cp, Th, Ea, Rg, k25, e_inf, x_max, t_max, 300, 'riviere')

    p, t = get_power_case0_river(T0, e0,  Rp, lamb, Dm/2, m, Cp, Th, Ea, Rg, k25, e_inf, x_max, t_max, 365)
    plot_powerXt(p, t)

#     Cas 1
    t, P_t = plot_power_vs_time_biocide(m=0.4,T0=25+273.15, Rp=5e-4, lamb_f=0.6, r=0.02, Cp=4184,Text=60+273.15, x_max=10,t_max=365,
                                        e0=2e-6, E=40e3, R=8.314, k25=1100, e_inf=100e-6,
                                        filename="Power_vs_time_biocide_river.pdf")
    


    m_choice, N_choice = plot_N_vs_m_biocide(
    scenario="river",
    T0=25+273.15, Rp=5e-4, lamb_f=0.6, r=0.02, Cp=4184,
    Text=60+273.15, x_max=10,
    t_max_short=365,
    e0=5e-6, E=40e3, R=8.314, k25=1100, e_inf=100e-6,
    P_target=20e6,
    filename="N_vs_m_biocide_mer.pdf" )

    C1r = pumping_cost(
    mdot=0.4,     # m3/s
    N=506,
    L=10.0,        # m
    D=0.02,        # m
    eta_pump=0.7,
    t_days=365)
    print(f"Coût annuel de pompage (rivière eau chaude) : {C1r:.0f} € / an")


#     Cas 2

    p, t,nc = get_power_case2_river(T0, e0, 900, Rp, lamb, Dm/2, 0.5, Cp, Th,Ea, Rg, k25, e_inf, x_max, t_max, 1000)
    print(nc)
    plot_powerXt(p, t)
    
    Nc = np.zeros((len(M), len(N)))
    for i, m in enumerate(M):
        for j, n in enumerate(N):
            Nc[i, j] = get_power_case2_river(T0, e0, n, Rp, lamb, Dm/2, m, Cp, Th,Ea, Rg, k25, e_inf, x_max, t_max, 1000)[2]  # Nc en MW
    N_grid, M_grid = np.meshgrid(N, M)
    plt.figure()
    plt.contourf(N_grid, M_grid, Nc, levels=20)
    plt.xlabel("Nombre de tubes n")
    plt.ylabel("Débit massique ṁ [kg/s]")
    plt.colorbar(label="M")
    save_figure('nombre_de_nettoyages_river', show=SHOW_PLOTS)
    
    C2r = pumping_cost(
    mdot=0.5,     # m3/s
    N=900,
    L=10.0,        # m
    D=0.02,        # m
    eta_pump=0.7,
    t_days=335)
    print(f"Coût annuel de pompage scenario 2 (mer) : {C2r:.0f} € / an")


    # Cas 3 
    p, t = get_power_case3(T0, e0,  Rp, lamb, Dm/2, m, Cp, Th, Ea, Rg, k25, e_inf, x_max, t_max, 3000, 'river')
    plot_powerXt(p, t)
    # Cas 4
    P=[]
    for n in N:
        p,t=get_power_case4(n,T0,0,Rp+Rc,lamb,(Dm-c)/2,m,Cp,Th,Ea,Rg,k25,e_inf,x_max,t_max,3000)
        P.append(p[-1])
    plt.ylabel('Puissance en MW')
    plt.xlabel('nombre de tubes')
    plt.grid()
    plt.plot(N,P)
    save_figure('puissance_vs_nombre_de_tubes_river', show=SHOW_PLOTS)


    C3r = pumping_cost(
    mdot=0.5,     # m3/s
    N=73,
    L=10.0,        # m
    D=0.02,        # m
    eta_pump=0.7,
    t_days=350)
    print(f"Coût annuel de pompage scenario 3 (rivière) : {C3r:.0f} € / an")


    #cas 4

    C4r = pumping_cost(
    mdot=0.4,     # m3/s
    N=1000,
    L=10.0,        # m
    D=0.02,        # m
    eta_pump=0.7,
    t_days=350)
    print(f"Coût annuel de pompage scenario 4 (rivière) : {C4r:.0f} € / an")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation, cases, or both")
    parser.add_argument("--simulation", action="store_true", help="Run simulation only")
    parser.add_argument("--cases", action="store_true", help="Run cases only")
    parser.add_argument("--no_show", action="store_true", help="Do not display plots")
    args = parser.parse_args()

    if args.no_show:
        globals()['SHOW_PLOTS'] = False

    run_simulation = args.simulation or (not args.simulation and not args.cases)
    run_cases = args.cases or (not args.simulation and not args.cases)

    if run_simulation:
        simulation()
    if run_cases:
        cases()
    