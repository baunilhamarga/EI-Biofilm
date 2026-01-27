import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import os

FIGURES_DIR = "figures"
SHOW_PLOTS = False

def save_figure(name, fig=None, show=False, pad_inches=0):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if fig is None:
        fig = plt.gcf()
    fig.savefig(os.path.join(FIGURES_DIR, f"{name}.pdf"), bbox_inches="tight", pad_inches=pad_inches)
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
    plt.ylabel(r'$e (\mu m)$')
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
    plt.ylabel(r'$e (\mu m)$')
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
    plt.ylabel(r'T $^oC$')
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
    plt.ylabel(r'T $^oC$')
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

    plt.ylabel(r'$T_f \> (°C)$')
    plt.xlabel('t (jours)')
    plt.legend(loc='best', fontsize="8")
    plt.grid()
    save_figure('plot_sensibility', show=SHOW_PLOTS)

# Plots de experimental data in addition to our model varying the
# parameters k25 and E
def plot_nebot(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf):
    # Nebot data
    R_nebot_file = open('r_nebot.txt', 'r')
    R_nebot = list(R_nebot_file.read().split('\n'))
    for i in range(len(R_nebot)): R_nebot[i] = float(R_nebot[i])

    time_nebot_file = open('time_nebot.txt', 'r')
    time_nebot = list(time_nebot_file.read().split('\n'))
    for i in range(len(time_nebot)): time_nebot[i] = float(time_nebot[i])

    # Nebot plot
    thickness = 2
    plt.plot(time_nebot, R_nebot, 's')

    # k25 tests
    list_k25 = [700, 1100, 1500]
    for current_k25 in list_k25:
        k25 = current_k25
        _, e, x, t  = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)
        Rb = e/(lamb_f * 1000)
        plt.plot(t, Rb[:, int(len(x)/2)], label=f'k25 = {k25}', linewidth=thickness)

    plt.ylabel(r'$R (m^2 K/kW)$')
    plt.xlabel('t (jours)')
    plt.legend(loc='center left', fontsize="8")
    plt.grid()
    save_figure('plot_nebot', show=SHOW_PLOTS)

    k25 = 1100
    plt.plot(time_nebot, R_nebot, 's')

    # Ea tests
    list_E = [30000, 40000, 50000]
    for current_E in list_E:
        E = current_E
        _, e, x, t  = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)
        Rb = e/(lamb_f * 1000)
        plt.plot(t, Rb[:, int(len(x)/2)], label=f'E = {E}', linewidth=thickness)

    plt.ylabel(r'$R (m^2 K/kW)$')
    plt.xlabel('t (jours)')
    plt.legend(loc='center left', fontsize="8")
    plt.grid()
    save_figure('plot_nebot_2', show=SHOW_PLOTS)




# Do a 3D plot of the behavior of the temperature in the system
def plotT3D(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf):
    T, _, x, t = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)
    x, t = np.meshgrid(x, t)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(t, x, T - 273.15, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
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
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    ax.set_xlabel('t (jours)')
    ax.set_ylabel('x (m)')
    ax.set_zlabel(r'$e\ (\mu m)$')

    save_figure('plote3D', fig, show=SHOW_PLOTS, pad_inches=0.15)

def power(N, m, Cp, Tf, Ti):
    return N*m*Cp*(Tf-Ti)

# Plots the temperature of the output of the pipes through time to different values of m
def plot_lastT_for_different_m(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density, water):
    thickness = 2

    m_list = [0.40, 0.45, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    for current_m in m_list:
        m = current_m
        T, _, _, t = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)
        plt.plot(t[1:], T[1:, -1] - 273.15, label=f'm = {m:.2f} kg/s', linewidth=thickness)
        Tf = T[-1, -1] - 273.15
        print(f'm = {m}, Tf = {Tf}')

    if water == 'sea':
        plt.axhline(27, linestyle='dashed')
    elif water == 'river':
        plt.axhline(30, linestyle='dashed')

    plt.ylabel(r'$T_f \> (ºC))$')
    plt.xlabel('t (jours)')
    plt.legend(loc='best', fontsize="8")
    plt.grid()
    save_figure('plot_lastT_for_different_m', show=SHOW_PLOTS)

# Get the power of the case 1 to sea scenario
def get_power_case1_sea(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density,):
    T, _, _, t = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)
    t = t[1:]
    Tf = T[1:, -1] - 273.15
    p = []

    Ti = T0 - 273.15
    
    for i in range(len(t)):
        p.append(power(679, m, Cp, Tf[i], Ti) * pow(10, -6))

    return p, t

# Get the power of the case 2 to sea scenario
def get_power_case2_sea(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density):
    T, _, _, t = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)
    t = t[1:]
    Tf = T[1:, -1] - 273.15
    p = []

    Ti = T0 - 273.15
    
    for i in range(len(t)):
        p.append(power(510, m, Cp, Tf[i], Ti) * pow(10, -6))

    return p, t

# Get the power of the case 3 to sea scenario
def get_power_case3_sea(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density):
    T, _, _, t = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)
    t = t[1:]
    Tf = T[1:, -1] - 273.15

    p = []
    Ti = T0 - 273.15
    cleaning = False
    n_cleans = 0
    for i in range(len(t)):
        current_power = power(550, m, Cp, Tf[i], Ti) * pow(10, -6)
        if current_power <= 22 and not cleaning:
            cleaning = True
            Ti_cleaning = t[i]
            n_cleans += 1

        if cleaning == True:
            current_power = 22

        if cleaning: 
            if t[i] - Ti_cleaning >= 1:
                cleaning = False
                j = i
                while j != len(Tf):
                    Tf[j] = Tf[j-i]
                    j += 1

        p.append(current_power)

    print(f'Number of cleans: {n_cleans}')

    return p, t

# Get the power of the case 3 to river scenario
def get_power_case3_river(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density):
    T, _, _, t = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)
    t = t[1:]
    Tf = T[1:, -1] - 273.15

    p = []
    Ti = T0 - 273.15
    cleaning = False
    n_cleans = 0
    for i in range(len(t)):
        current_power = power(760, m, Cp, Tf[i], Ti) * pow(10, -6)
        if current_power <= 22 and not cleaning:
            cleaning = True
            Ti_cleaning = t[i]
            n_cleans += 1

        if cleaning == True:
            current_power = 22

        if cleaning: 
            if t[i] - Ti_cleaning >= 1:
                cleaning = False
                j = i
                while j != len(Tf):
                    Tf[j] = Tf[j-i]
                    j += 1

        p.append(current_power)

    print(f'Number of cleans: {n_cleans}')

    return p, t

# Get the power of the case 4
def get_power_case4(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density, water):
    if water == 'sea':
        n_pipes = 550
        frequency = 2 * (density/365)
        n_groups = 11
    if water == 'river':
        n_pipes = 760
        frequency = 2 * (density/365)
        n_groups = 15
    
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

    print(f'Number of cleans: {n_cleans}')

    return p, t

# Get the power of the case 1 to river scenario
def get_power_case1_river(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density):
    T, _, _, t = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)
    t = t[1:]
    Tf = T[1:, -1] - 273.15
    p = []

    Ti = T0 - 273.15
    
    for i in range(len(t)):
        p.append(power(933, m, Cp, Tf[i], Ti) * pow(10, -6))

    return p, t

# Get the power of the case 2 to river scenario
def get_power_case2_river(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density):
    T, _, _, t = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)
    t = t[1:]
    Tf = T[1:, -1] - 273.15
    p = []

    Ti = T0 - 273.15
    
    for i in range(len(t)):
        p.append(power(700, m, Cp, Tf[i], Ti) * pow(10, -6))

    return p, t

# Plots the power over time
def plot_powerXt(p, t):
    thickness = 2

    plt.plot(t, p, linewidth=thickness)
    plt.axhline(22, color='red', linestyle='dashed')
    plt.ylabel(r'$Puissance \> (MW)$')
    plt.xlabel('t (jours)')
    plt.grid()
    save_figure('plot_powerXt', show=SHOW_PLOTS)

# Plots the temperature of the output of the pipes through time to different values of e_inf
def plot_lastT_diferent_einf(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density):
    thickness = 2

    percentages = [1, 0.3]
    for percentage in percentages:
        T, _, x, t = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)
        t_plot = t
        x, t = np.meshgrid(x, t)
        percentage_to_plot = percentage * 100
        plt.plot(t_plot[1:], T[1:, -1] - 273.15, label=f'{percentage_to_plot}% de e_inf', linewidth=thickness)
        Tf = T[-1, -1]-273.15
        print(f'Percentage = {percentage}, Tf = {Tf}, m = {m}')

    plt.ylabel(r'$T_f \> (ºC)$')
    plt.xlabel('t (jours)')
    plt.legend(loc='best', fontsize="8")
    plt.grid()
    save_figure('plot_lastT_diferent_einf', show=SHOW_PLOTS)

def plot_power_npipes(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density):
    T, _, _, t = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)
    t = t[1:]
    Tf = T[1:, -1] - 273.15
    thickness = 2

    Ti = T0 - 273.15
    n_list = np.arange(600, 1000, int((1000-600)/10))
    p = []
    for current_n in n_list:
        for i in range(len(t)):
            p.append(power(current_n, m, Cp, Tf[i], Ti) * pow(10, -6))
        plt.plot(t, p, linewidth=thickness, label = f'{current_n} tuyaux')
        print(f'{current_n} tuyaux: Pmax = {p[0]}, Pmin = {p[-1]}')
        p = []

    plt.axhline(22, color='red', linestyle='dashed')
    plt.ylabel(r'$Puissance \> (MW)$')
    plt.xlabel('t (jours)')
    plt.legend(loc='best', fontsize="8")
    plt.grid()
    save_figure('plot_power_npipes', show=SHOW_PLOTS)

def case1_npipesXm(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density):
    m_list = np.arange(0.4, 0.8, (0.8-0.45)/10)
    n_tubes = []
    for current_m in m_list:
        m = current_m
        T, _, _, _ = simulate_T_e(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)
        Ti = T0 - 273.15
        Tf = T[-1, -1] - 273.15
        n_tubes.append((22*pow(10, 6)) / (current_m*Cp*(Tf - Ti)))

    plt.ylabel(r'$\text{N tuyaux}$')
    plt.xlabel(r'$\dot{m} \> kg/s$')
    plt.plot(m_list, n_tubes)
    plt.grid()
    save_figure('case1_npipesXm', show=SHOW_PLOTS)

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

    # Initial conditions
    T0 = 17 + 273.15         # [K]
    e0 = 0.01 * e_inf        # [m]

    plotT3D(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)
    plote3D(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)
    plot_sensibility(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)
    plot_nebot(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)
    plot_eXt_eXx(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)
    plot_TXt_TXx(T0, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf)

# def cases():
#     # Contants
#     Dm = 0.02                    # [m]
#     x_max = 10                   # [m]
#     Th = 60 + 273.15             # [K]
#     Cp = 4184                    # [J/kg/K]
#     lamb = 0.6                   # [W/m/K]
#     Rp = 5e-4                    # [m^2 * K / W]
#     Ea = 40000                   # [J/mol]
#     Rg = 8.314                   # [J/(mol * K)]
#     k25 = 1100
#     Ar = (math.pi * pow(Dm, 2)) /4
#     e_inf = 200e-6
#     t_max = 365
#     e0 = 0.01*e_inf

#     # Parameters sea
#     m = 0.8
#     T0 = 12 + 273.15
#     v = m/(1000 * Ar)
    
#     #print('-------Seawater----------')
#     # Case 1 679
#     plot_lastT_for_different_m(T0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, x_max, t_max, 300, 'sea')
#     p, t = get_power_case1_sea(T0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, x_max, t_max, 1000)
#     plot_powerXt(p, t)
#     case1_npipesXm(T0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, x_max, t_max, 300)

#     # Case 2 510
#     plot_lastT_diferent_einf(T0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, x_max, t_max, 1000)
#     p, t = get_power_case2_sea(T0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf*0.3, x_max, t_max, 1000)
#     plot_powerXt(p, t)

#     # Case 3: 550 pipes
#     plot_power_npipes(T0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, x_max, t_max, 300)
#     p, t = get_power_case3_sea(T0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, x_max, t_max, 300)
#     plot_powerXt(p, t)

#     # Case 4: 550
#     p, t = get_power_case4(T0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, x_max, t_max, 365*2, 'sea')
#     plot_powerXt(p, t)

#     #print('-------River Water----------')
#     # Parameters river
#     m = 0.8
#     T0 = 25 + 273.15
#     v = m/(1000 * Ar)

#     # Case 1 933 pipes
#     plot_lastT_for_different_m(T0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, x_max, t_max, 365, 'river')
#     p, t = get_power_case1_river(T0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, x_max, t_max, 356)
#     plot_powerXt(p, t)

#     # Case 2 700 pipes
#     plot_lastT_diferent_einf(T0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, x_max, t_max, 1000)
#     p, t = get_power_case2_river(T0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf*0.3, x_max, t_max, 365)
#     plot_powerXt(p, t)

#     # Case 3 760 pipes
#     plot_power_npipes(T0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, x_max, t_max, 300)
#     p, t = get_power_case3_river(T0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, x_max, t_max, 365)
#     plot_powerXt(p, t)

#     # Case 4 760 pipes
#     p, t = get_power_case4(T0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, x_max, t_max, 365*2, 'river')
#     plot_powerXt(p, t)

if __name__ == "__main__":
    simulation()
    # cases()
