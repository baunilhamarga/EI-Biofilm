import numpy as np
import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# Rtot value
def Rtot(e, lamb, Rp):
    Rtot = e / lamb + Rp
    return Rtot

# dTdx value
def dTdx(T, e, lamb_f, Rp, r, m, Cp, Text):
    Rtot = Rtot(e, lamb_f, Rp)
    dTdx = (2 * math.pi * r) / (m * Cp * Rtot) * (Text - T)
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
def simulate_T_e(T, T0, e, e0, Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max = 3.1, t_max = 60, density = 1000):
    dx = x_max/density
    x = np.arange(0, x_max, dx)

    dt = t_max/density
    t = np.arange(0, t_max, dt)

    # Tc: time x position
    T = [[T0 for _ in range(len(x))] for _ in range(len(t))]
    e = [[e0 for _ in range(len(x))] for _ in range(len(t))]

    for i in range(1, len(t)):
        for j in range(1, len(x)):
            T[i][j] = (T[i][j-1] + dTdx(T[i][j-1], e[i][j-1], T, e, lamb_f, Rp, r, m, Cp, Text) * dx)
            e[i][j] = (e[i-1][j] + dedt(T[i-1][j], e[i-1][j], lamb_f, E, R, k25, e_inf) * dt)

    T = np.array(T)
    e = np.array(e) * pow(10, 6)

    return T, e, x, t    

# Plots e in function of x for fixed values of time
def plot_e_x(e, t, x):
    thickness = 2

    percentages = [5/50, 10/50, 15/50, 20/50, 25/50, 30/50, 35/50, 40/50, 49.9/50]
    days = [5, 10, 15, 20, 25, 30, 35, 40, 50]

    for i in range(len(percentages)):
        plt.plot(x[1:], e[int(len(t)*percentages[i]), 1:], label=f'{days[i]} jours', linewidth=thickness)
    plt.ylabel(r'$e (\mu m)$')
    plt.xlabel('x(m)')
    plt.legend(loc='center left', fontsize="8")
    plt.grid()
    plt.show()

# Plots e in function of time for fixed percentages of total lenght of the pipe
def plot_e_t(e, t, x):
    thickness = 2

    percentages = [0.25, 0.5, 0.75, 0.99]
    percent = [25, 50, 75, 100]

    for i in range(len(percentages)):
        plt.plot(t, e[:, int(len(x) * percentages[i])], label=f'{percent[i]}%', linewidth=thickness)
    plt.ylabel(r'$e (\mu m)$')
    plt.xlabel('t (jours)')
    plt.legend(loc='center left', fontsize="8")
    plt.grid()
    plt.show()

# Plots both eXt and eXx
def plot_eXt_eXx(T, T0, e, e0, Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max = 3.1, t_max = 60, density = 1000):
    T, e, x, t = simulate_T_e(T, T0, e, e0, Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density)

    plot_e_t(T, e, t, x)
    plot_e_x(T, e, t, x)

# Give different plots of e in function of x in the last instant of time for different
# numbers of points to use in simulation
def plot_sensibility(T, T0, e, e0,  Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max = 3.1, t_max = 60):
    thickness = 2

    densities = [50, 100, 150, 300, 500, 750, 1000]
    for density in densities:
        T, e, x, t = simulate_T_e(T, T0, e, e0, Rp, lamb_f, r, m, Cp, Text, E, R, k25, e_inf, x_max, t_max, density = density)
        t_plot = t
        x, t = np.meshgrid(x, t)
        plt.plot(t_plot[1:], T[1:, -1] - 273.15, label=f'{density} points', linewidth=thickness)

    plt.ylabel(r'$T_f \> (ºC)$')
    plt.xlabel('t (jours)')
    plt.legend(loc='best', fontsize="8")
    plt.grid()
    plt.show()

# Plots de experimental data in addition to our model varying the
# parameters k25 and E
def plot_nebot(Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf):
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
        Tc, e, z, t  = simulate_Tc_e_with_initial(Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf)
        Rb = e/(lamb * 1000)
        plt.plot(t, Rb[:, int(len(z)/2)], label=f'k25 = {k25}', linewidth=thickness)

    plt.ylabel(r'$R (m^2 K/kW)$')
    plt.xlabel('t (jours)')
    plt.legend(loc='center left', fontsize="8")
    plt.grid()
    plt.show()

    k25 = 1100
    plt.plot(time_nebot, R_nebot, 's')

    # Ea tests
    list_Ea = [30000, 40000, 50000]
    for current_Ea in list_Ea:
        Ea = current_Ea
        Tc, e, z, t  = simulate_Tc_e_with_initial(Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf)
        Rb = e/(lamb * 1000)
        plt.plot(t, Rb[:, int(len(z)/2)], label=f'E = {Ea}', linewidth=thickness)

    plt.ylabel(r'$R (m^2 K/kW)$')
    plt.xlabel('t (jours)')
    plt.legend(loc='center left', fontsize="8")
    plt.grid()
    plt.show()

# Do a 3D plot of the behavior of the system
def plot3D(Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf):
    Tc, e, z, t = simulate_Tc_e_with_initial(Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf)
    z, t = np.meshgrid(z, t)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(t, z, Tc - 273.15, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    ax.set_xlabel('t (jours)')
    ax.set_ylabel('z (m)')
    ax.set_zlabel(r'T $^oC$')

    plt.show()

def power(N, m, Cp, Tf, Ti):
    return N*m*Cp*(Tf-Ti)

# Simulates Tc and e in function of time and .
# Returns the linear space of time and z
# and the matrices Tc and z
def simulate_Tc_e_with_initial(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density):
    dz = z_max/density
    z = np.arange(0, z_max, dz)

    dt = t_max/density
    t = np.arange(0, t_max, dt)

    # Tc: time x position
    Tc = [[Tc0 for _ in range(len(z))] for _ in range(len(t))]
    e = [[e0 for _ in range(len(z))] for _ in range(len(t))]

    for i in range(1, len(t)):
        for j in range(1, len(z)):
            Tc[i][j] = (Tc[i][j-1] + dTdx(e[i][j-1], Tc[i][j-1], Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf) * dz)
            e[i][j] = (e[i-1][j] + dedt(e[i-1][j], Tc[i-1][j], Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf) * dt)

    Tc = np.array(Tc)
    e = np.array(e) * pow(10, 6)

    return Tc, e, z, t

# Plots the temperature of the output of the pipes through time to different values of m
def plot_lastTc_for_different_m(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density, water):
    thickness = 2

    m_list = [0.40, 0.45, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    for current_m in m_list:
        m = current_m
        Tc, e, z, t = simulate_Tc_e_with_initial(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density)
        plt.plot(t[1:], Tc[1:, -1] - 273.15, label=f'm = {m:.2f} kg/s', linewidth=thickness)
        Tf = Tc[-1, -1] - 273.15
        print(f'm = {m}, Tf = {Tf}')

    if water == 'sea':
        plt.axhline(27, linestyle='dashed')
    elif water == 'river':
        plt.axhline(30, linestyle='dashed')

    plt.ylabel(r'$T_f \> (ºC))$')
    plt.xlabel('t (jours)')
    plt.legend(loc='best', fontsize="8")
    plt.grid()
    plt.show()

# Get the power of the case 1 to sea scenario
def get_power_case1_sea(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density):
    Tc, e, z, t = simulate_Tc_e_with_initial(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density)
    t = t[1:]
    Tf = Tc[1:, -1] - 273.15
    p = []

    Ti = Tc0 - 273.15
    
    for i in range(len(t)):
        p.append(power(679, m, Cp, Tf[i], Ti) * pow(10, -6))

    return p, t

# Get the power of the case 2 to sea scenario
def get_power_case2_sea(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density):
    Tc, e, z, t = simulate_Tc_e_with_initial(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density)
    t = t[1:]
    Tf = Tc[1:, -1] - 273.15
    p = []

    Ti = Tc0 - 273.15
    
    for i in range(len(t)):
        p.append(power(510, m, Cp, Tf[i], Ti) * pow(10, -6))

    return p, t

# Get the power of the case 3 to sea scenario
def get_power_case3_sea(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density):
    Tc, e, z, t = simulate_Tc_e_with_initial(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density)
    t = t[1:]
    Tf = Tc[1:, -1] - 273.15

    p = []
    Ti = Tc0 - 273.15
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
def get_power_case3_river(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density):
    Tc, e, z, t = simulate_Tc_e_with_initial(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density)
    t = t[1:]
    Tf = Tc[1:, -1] - 273.15

    p = []
    Ti = Tc0 - 273.15
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
def get_power_case4(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density, water):
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
    Tc, e, z, t = simulate_Tc_e_with_initial(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density)
    Tfs_reference = Tc[1:, -1] - 273.15
    for i in range(n_groups):
        Tc, e, z, t = simulate_Tc_e_with_initial(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density)
        Tfs.append(Tc[1:, -1] - 273.15)
        Ti_cleaning.append(0)
        cleaning.append(False)
    t = t[1:]

    p = []
    Ti = Tc0 - 273.15
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
def get_power_case1_river(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density):
    Tc, e, z, t = simulate_Tc_e_with_initial(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density)
    t = t[1:]
    Tf = Tc[1:, -1] - 273.15
    p = []

    Ti = Tc0 - 273.15
    
    for i in range(len(t)):
        p.append(power(933, m, Cp, Tf[i], Ti) * pow(10, -6))

    return p, t

# Get the power of the case 2 to river scenario
def get_power_case2_river(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density):
    Tc, e, z, t = simulate_Tc_e_with_initial(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density)
    t = t[1:]
    Tf = Tc[1:, -1] - 273.15
    p = []

    Ti = Tc0 - 273.15
    
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
    plt.show()

# Plots the temperature of the output of the pipes through time to different values of e_inf
def plot_lastTc_diferent_einf(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density):
    thickness = 2

    percentages = [1, 0.3]
    for percentage in percentages:
        Tc, e, z, t = simulate_Tc_e_with_initial(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf*percentage, z_max, t_max, density)
        t_plot = t
        z, t = np.meshgrid(z, t)
        percentage_to_plot = percentage * 100
        plt.plot(t_plot[1:], Tc[1:, -1] - 273.15, label=f'{percentage_to_plot}% de e_inf', linewidth=thickness)
        Tf = Tc[-1, -1]-273.15
        print(f'Percentage = {percentage}, Tf = {Tf}, m = {m}')

    plt.ylabel(r'$T_f \> (ºC)$')
    plt.xlabel('t (jours)')
    plt.legend(loc='best', fontsize="8")
    plt.grid()
    plt.show()

def plot_power_npipes(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density):
    Tc, e, z, t = simulate_Tc_e_with_initial(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density)
    t = t[1:]
    Tf = Tc[1:, -1] - 273.15
    thickness = 2

    Ti = Tc0 - 273.15
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
    plt.show()

def case1_npipesXm(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density):
    thickness = 2

    m_list = np.arange(0.4, 0.8, (0.8-0.45)/10)
    n_tubes = []
    for current_m in m_list:
        m = current_m
        Tc, e, z, t = simulate_Tc_e_with_initial(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, density)
        Ti = Tc0 - 273.15
        Tf = Tc[-1, -1] - 273.15
        n_tubes.append((22*pow(10, 6)) / (current_m*Cp*(Tf - Ti)))

    plt.ylabel(r'$\text{N tuyaux}$')
    plt.xlabel(r'$\dot{m} \> kg/s$')
    plt.plot(m_list, n_tubes)
    plt.grid()
    plt.show()

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
    Dm = 0.012                   # [m]
    Th = 35 + 273.15             # [K]
    Cp = 4184                    # [J/kg/K]
    lamb = 0.6                   # [W/m/K]
    Rp = 5e-4                    # [m^2 * K / W]
    Ea = 40000                   # [J/mol]
    Rg = 8.314                   # [J/(mol * K)]
    k25 = 1100
    Ar = (math.pi * pow(Dm, 2)) /4
    v = 1.85                     # [m/s]
    m = 1000 * Ar * v            # [kg/s]
    e_inf = 200e-6

    plot3D(Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf)
    plot_sensibility(Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf)
    plot_nebot(Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf)
    plot_eXt_eXz(Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf)

def cases():
    # Contants
    Dm = 0.02                    # [m]
    z_max = 10                   # [m]
    Th = 60 + 273.15             # [K]
    Cp = 4184                    # [J/kg/K]
    lamb = 0.6                   # [W/m/K]
    Rp = 5e-4                    # [m^2 * K / W]
    Ea = 40000                   # [J/mol]
    Rg = 8.314                   # [J/(mol * K)]
    k25 = 1100
    Ar = (math.pi * pow(Dm, 2)) /4
    e_inf = 200e-6
    t_max = 365
    e0 = 0.01*e_inf

    # Parameters sea
    m = 0.8
    Tc0 = 12 + 273.15
    v = m/(1000 * Ar)
    
    #print('-------Seawater----------')
    # Case 1 679
    plot_lastTc_for_different_m(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, 300, 'sea')
    p, t = get_power_case1_sea(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, 1000)
    plot_powerXt(p, t)
    case1_npipesXm(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, 300)

    # Case 2 510
    plot_lastTc_diferent_einf(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, 1000)
    p, t = get_power_case2_sea(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf*0.3, z_max, t_max, 1000)
    plot_powerXt(p, t)

    # Case 3: 550 pipes
    plot_power_npipes(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, 300)
    p, t = get_power_case3_sea(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, 300)
    plot_powerXt(p, t)

    # Case 4: 550
    p, t = get_power_case4(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, 365*2, 'sea')
    plot_powerXt(p, t)

    #print('-------River Water----------')
    # Parameters river
    m = 0.8
    Tc0 = 25 + 273.15
    v = m/(1000 * Ar)

    # Case 1 933 pipes
    plot_lastTc_for_different_m(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, 365, 'river')
    p, t = get_power_case1_river(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, 356)
    plot_powerXt(p, t)

    # Case 2 700 pipes
    plot_lastTc_diferent_einf(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, 1000)
    p, t = get_power_case2_river(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf*0.3, z_max, t_max, 365)
    plot_powerXt(p, t)

    # Case 3 760 pipes
    plot_power_npipes(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, 300)
    p, t = get_power_case3_river(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, 365)
    plot_powerXt(p, t)

    # Case 4 760 pipes
    p, t = get_power_case4(Tc0, e0, Dm, Th, Cp, lamb, Rp, Ea, Rg, k25, Ar, v, m, e_inf, z_max, t_max, 365*2, 'river')
    plot_powerXt(p, t)

if __name__ == "__main__":
    simulation()
    cases()