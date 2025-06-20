import cvxpy as cp
import time
import os
import pandas as pd
import numpy as np
from Plot_Results import (print_hourly_linepack, print_hourly_demand, print_pressure_table, print_hourly_demand_elec,
                        print_errors_table, print_flow_table)
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
a = 1

print("CHECK POINT : ** code initiation **")

# DATA
filename = "Data.xlsx"
sheet_name = '12-2024'
day = 3
µ = 0.5

# DATA SHOULD BE CONVERTED INTO PU !!
base = 100  # Base of 100 MVA used
T = 24

E_elec = 5  # number of conventional productions
N_elec = 14  # NODES
L_elec = 20  # LINES
W_elec = 2  # Wind farms

N_gas = 12  # Number of gas node
L_gas = 12  # Number of gas pipeline
W_gas = 3  # Number of gas wells

"**********************************************************************************"
"                                      Elec Data                                   "
"**********************************************************************************"

ElecDemand = pd.read_excel(filename, sheet_name="DemandData", usecols=["EL demand (MW)"])  # In MWh
ElecDemand = (ElecDemand.iloc[:, 0].to_numpy())/base  # ndarray(24) in pu

DemandShare = pd.read_excel(filename, sheet_name="DemandData", usecols=["Share (%)"]).dropna()  # In %
DemandShare = DemandShare.iloc[:, 0].to_numpy()  # ndarray(14) in %

Pmax = pd.read_excel(filename, sheet_name="PowerUnitData", usecols=["Pmax"])
Pmax = (Pmax.iloc[:, 0].to_numpy())/base  # ndarray(5) in pu
Pmin = pd.read_excel(filename, sheet_name="PowerUnitData", usecols=["Pmin"])
Pmin = (Pmin.iloc[:, 0].to_numpy())/base  # ndarray(5) in pu

Pwind = pd.read_excel(filename, sheet_name="WindData", usecols=["Pmax (MW)"]).dropna()
Pwind = (Pwind.iloc[:, 0].to_numpy())/base  # ndarray(2) in pu

b_line = pd.read_excel(filename, sheet_name="ElNetwork", usecols=["b"])  # Line susceptance
b_line = b_line.iloc[:, 0].to_numpy()  # ndarray(20)  already in pu

lines = pd.read_excel(filename, sheet_name="ElNetwork", usecols=[1,2])  # Line with node numbers
lines = lines.iloc[:, :].to_numpy()

LineFrom = np.array([[l for l in range(L_elec) if lines[l, 0] == n + 1] for n in range(N_elec)], dtype=object)
LineTo = np.array([[l for l in range(L_elec) if lines[l, 1] == n + 1] for n in range(N_elec)], dtype=object)

Fmax = pd.read_excel(filename, sheet_name="ElNetwork", usecols=["Fmax"])  # Transmission line capacity limits
Fmax = (Fmax.iloc[:, 0].to_numpy()) / base

Emap = pd.read_excel(filename, sheet_name="PowerUnitMap")  # Assign the location of the production to nodes
Emap = Emap.iloc[:, :].to_numpy()

Wmap = pd.read_excel(filename, sheet_name="WindMap")  # Assign the location of the production to nodes
Wmap = Wmap.iloc[:, :].to_numpy()

Costs = pd.read_excel(filename, sheet_name="Costs").dropna()
Efficiency = Costs.loc[:, "Efficiency [Mwhe/MWhf]"].to_numpy()  # Nucl : [0]  GFPP : [1]
Fuel_cost = Costs.loc[:, "Fuel cost [€/MWhf]"].to_numpy()
VOM = Costs.loc[:, "VOM[€/MWh]"].to_numpy()
CO2 = Costs.loc[:, "Carbon emission [tCO2/MWhe]"].to_numpy()
CO2_price = Costs.loc[1, "Carbon price [€/tCO2]"]

"**********************************************************************************"
"                                      Gas Data                                   "
"**********************************************************************************"

ConversionFactor = 11200  # MWh/MNm3 of gas
ramp_up_limit = 0.2  # 10% of max output
ramp_down_limit = 0.2

Gmax = pd.read_excel(filename, sheet_name="GasWellData", usecols=["Gmax (Mm3)"])
Gmax = (Gmax.iloc[:, 0].to_numpy())  # ndarray(3) in Mm3

c_gas = pd.read_excel(filename, sheet_name="GasWellData", usecols=["Day-ahead offer price (€/Mm3)"])
c_gas = (c_gas.iloc[:, 0].to_numpy())  # ndarray(3) in €/Mm3

Prmin = pd.read_excel(filename, sheet_name="GasNetwork", usecols=["Min pressure (bar)"])
Prmin = (Prmin.iloc[:, 0].to_numpy())  # ndarray(12) for each gas node in bar

Prmax = pd.read_excel(filename, sheet_name="GasNetwork", usecols=["Max pressure (bar)"])
Prmax = (Prmax.iloc[:, 0].to_numpy())  # ndarray(12) for each gas node in bar

WellMap = pd.read_excel(filename, sheet_name="GasWellMap")  # Assign the location of the gas wells
WellMap = WellMap.iloc[:, :].to_numpy()

GasDemandShare = pd.read_excel(filename, sheet_name="DemandData", usecols=["Gas Share (%)"]).dropna()  # In %
GasDemandShare = GasDemandShare.iloc[:, 0].to_numpy()  # ndarray(12) in %

pipes = pd.read_excel(filename, sheet_name="GasNetwork", usecols=[1,2])  # pipelines with node numbers
pipes = pipes.iloc[:, :].to_numpy()

PipeFrom = np.array([[l for l in range(L_gas) if pipes[l, 0] == n + 1] for n in range(N_gas)], dtype=object)  # Pipeline starting at this node
PipeTo = np.array([[l for l in range(L_gas) if pipes[l, 1] == n + 1] for n in range(N_gas)], dtype=object)  # Pipeline ending at this node

H_init = pd.read_excel(filename, sheet_name="GasNetwork", usecols=["Initial linepack (Mm3)"])  # Initial quantity of gas stored in a pipeline
H_init = H_init.iloc[:, 0].to_numpy()

K_h = pd.read_excel(filename, sheet_name="GasNetwork", usecols=["Linepack constant (Mm3/bar)"])  # Quantifies the gas volume added/removed for each pressure change
K_h = K_h.iloc[:, 0].to_numpy()  # ndarray(12) in [Mm3/bar]

GFPPmap = pd.read_excel(filename, sheet_name="GFPPMap")  # Assign the location of the gas fired power plants
GFPPmap = GFPPmap.iloc[:, :].to_numpy()

C_ratio = pd.read_excel(filename, sheet_name="GasNetwork", usecols=["Compressor factor"])
C_ratio = (C_ratio.iloc[:, 0].to_numpy())

K_w = pd.read_excel(filename, sheet_name="GasNetwork", usecols=["Natural gas flow constant (Mm3/bar)"])
K_w = K_w.iloc[:, 0].to_numpy()  # ndarray(12) in [Mm3/bar]

"**********************************************************************************"
"                                     Daily Data                                   "
"**********************************************************************************"

gasfile = "GasConsumptionDistrib.xlsx"

GasDemand = pd.read_excel(gasfile, sheet_name=sheet_name, usecols=["Physical Flow Rescaled (MNm3)"])  # In Mm3
GasDemand = (GasDemand.iloc[:, 0].to_numpy())  # ndarray(24) in Mm3

# Dynamically determine the number of days (assume 24 rows per day)
n_hours = 24
n_days = len(GasDemand) // n_hours  # Calculate number of days
assert len(GasDemand) % n_hours == 0, "The data does not align perfectly into 24-hour periods."
GasDemand = GasDemand.reshape(n_days, n_hours)

windfile = "WindForecast.xlsx"

WindFactor = pd.read_excel(windfile, sheet_name=sheet_name, usecols=["Wind factor [%]"])
WindFactor = WindFactor.iloc[:, 0].to_numpy()  # ndarray(24) in %

# Dynamically determine the number of days (assume 24 rows per day)
n_hours = 24
n_days = len(WindFactor) // n_hours  # Calculate number of days
assert len(WindFactor) % n_hours == 0, "The data does not align perfectly into 24-hour periods."
WindFactor = WindFactor.reshape(n_days, n_hours)

print("CHECK POINT : ** data loaded **")

"**********************************************************************************"
"                               Optimization problem                               "
"**********************************************************************************"
tic = time.perf_counter()
# Define optimization variables
# Electricity network
p = cp.Variable((E_elec, T), nonneg=True)  # Electrical power produced by each conventional unit [MWh]
theta = cp.Variable((N_elec, T))
f = cp.Variable((L_elec, T))

# Gas network
g = cp.Variable((W_gas, T), nonneg=True)  # Gas volume bought to each gas well [Mm3]
pr = cp.Variable((N_gas, T), nonneg=True)  # Pressure at each node of the gas grid [bar]
q = cp.Variable((L_gas, T), nonneg=True)  # Average gas flow in a pipeline [MNm3]
q_in = cp.Variable((L_gas, T), nonneg=True)  # Gas entering the pipeline [MNm3]
q_out = cp.Variable((L_gas, T), nonneg=True)  # Gas exiting the pipeline [MNm3]
h = cp.Variable((L_gas, T), nonneg=True)  # Linepack in a pipeline [MNm3]
ε_q = cp.Variable((L_gas, T), nonneg=True)

# Parameters
GasDemandP = cp.Parameter(24, nonneg=True)
WindFactorP = cp.Parameter(24, nonneg=True)

# WEYMOUTH PARAMETERS
PR = cp.Parameter((N_gas, T), nonneg=True)
F0 = cp.Parameter((L_gas, T), nonneg=True)

# Define constraints
# Elec constraints
constraints = []

for t in range(T):

    ref_node = theta[0, t] == 0
    theta_minus = theta[:, t] >= -np.pi/6
    theta_plus = theta[:, t] <= np.pi/6

    power_limits_min = p[:, t] >= Pmin[:]
    power_limits_max = p[:, t] <= Pmax[:]

    constraints += [ref_node, theta_minus, theta_plus, power_limits_min, power_limits_max]

    for l in range(L_elec):
        flows = f[l, t] == b_line[l] * (theta[lines[l, 0]-1, t] - theta[lines[l, 1]-1, t])  # Need a -1 bc indexing start at 0
        line_limit_up = f[l, t] <= Fmax[l]
        line_limit_down = f[l, t] >= -Fmax[l]
        constraints += [flows, line_limit_up, line_limit_down]

    for n in range(N_elec):
        LT = LineTo[n]
        LF = LineFrom[n]
        nodal_balance = (Emap[n, :] @ p[:, t] + cp.sum([f[l, t] for l in LT]) + Wmap[n, :] @ Pwind[:] * WindFactorP[t]
                         == DemandShare[n]*ElecDemand[t] + cp.sum([f[l, t] for l in LF]))
        constraints += [nodal_balance]

# Gas constraints

for t in range(T):

    gas_limits_max = g[:, t] <= Gmax[:]
    pr_limits_min = pr[:, t] >= Prmin[:]
    pr_limits_max = pr[:, t] <= Prmax[:]

    if 0 < t < 23:
        ramp_up = g[:, t] - g[:, t-1] <= ramp_up_limit*Gmax[:]
        ramp_down = g[:, t-1] - g[:, t] <= ramp_down_limit*Gmax[:]
        constraints += [ramp_up, ramp_down]

    for l in range(L_gas):
        node1 = pipes[l, 0]
        node2 = pipes[l, 1]
        if C_ratio[l] != 1:
            compressor1 = pr[node2 - 1, t] <= C_ratio[l] * pr[node1 - 1, t]
            compressor2 = pr[node1 - 1, t] <= pr[node2 - 1, t]
            constraints += [compressor1, compressor2]
        else:
            compressor2 = pr[node2 - 1, t] <= pr[node1 - 1, t]
            constraints += [compressor2]

    AvFlow = q[:, t] == (q_in[:, t] + q_out[:, t]) / 2

    if t == 0:
        linepack1 = h[:, t] == H_init[:] + q_in[:, t] - q_out[:, t]
    else:
        linepack1 = h[:, t] == h[:, t-1] + q_in[:, t] - q_out[:, t]
    if t == 23:
        linepack_final = cp.sum([h[l, t] for l in range(L_gas)]) == cp.sum([H_init[l] for l in range(L_gas)])

        constraints += [linepack_final]

    constraints += [gas_limits_max, pr_limits_min, pr_limits_max, AvFlow, linepack1]

    for l in range(L_gas):
        node1 = pipes[l, 0]
        node2 = pipes[l, 1]
        linepack2 = h[l, t] == K_h[l] * (pr[node1 - 1, t] + pr[node2 - 1, t]) / 2

        constraints += [linepack2]

    for n in range(N_gas):
        PT = PipeTo[n]
        PF = PipeFrom[n]

        nodal_gas_balance = (WellMap[n, :] @ g[:, t] + cp.sum([q_out[l, t] for l in PT]) ==
                            (GFPPmap[n, :] @ p[:, t])*base / (ConversionFactor*Efficiency[0]) + GasDemandShare[n]*(GasDemandP[t])
                             + cp.sum([q_in[l, t] for l in PF]))

        constraints += [nodal_gas_balance]

    # ___________ !! Weymouth !!______________ #

    for l in range(L_gas):

        if l == 1 or l == 8:  # Skip pipeline if presence of a compressor
            continue
        node1 = pipes[l, 0]
        node2 = pipes[l, 1]

        A = PR[node1-1, t] / F0[l, t]
        B = - PR[node2-1, t] / F0[l, t]
        weymouth = q[l, t] == K_w[l] * (F0[l, t] + A * (pr[node1-1, t]-PR[node1-1, t])
                              + B * (pr[node2-1, t]-PR[node2-1, t]))

        # Trust Region Constraints

        trust_region1 = (q[l, t] - (K_w[l] * F0[l, t]) >= -ε_q)
        trust_region2 = (q[l, t] - (K_w[l] * F0[l, t]) <= ε_q)

        constraints += [weymouth, trust_region1, trust_region2]


# Define objective function

variable_elec_cost = (Fuel_cost[1]/Efficiency[1]) + (CO2[1] * CO2_price) + VOM[1]
elec_cost = cp.sum(p * variable_elec_cost)
error_cost = cp.sum(ε_q)
gas_cost = cp.sum(c_gas @ g)

#objective = cp.Minimize(elec_cost + gas_cost)
#objective = cp.Minimize(elec_cost + gas_cost + 1000000*error_cost)
objective = cp.Minimize(((elec_cost + gas_cost)*(1-µ)) + (1000000*error_cost*µ))

# Define the problem
prob = cp.Problem(objective, constraints)

# Load the NN predictor
model_path = "model_Enco_Deco_full.keras"
pred_model = load_model(model_path)

NMAE_tot = []
MAE_tot = []
STD_tot = []
cost_tot = []

for day in range(3, n_days):

    # -----------------------------------------NEURAL NETWORK PREDICTION ---------------------------------------------

    gas_flat = GasDemand.flatten()  # Shape (744,)
    wind_flat = WindFactor.flatten()  # Shape (744,)
    start_index = (day-3) * 24
    gas_72 = gas_flat[start_index:start_index + 72]  # Shape (72,)
    wind_72 = wind_flat[start_index:start_index + 72]  # Shape (72,)
    final_array = np.stack((gas_72, wind_72), axis=1)  # Shape (72,2)
    x = np.expand_dims(final_array, axis=0)  # Shape (1,72,2)

    training_mean_x = [0.19187630601305095, 0.2615116389106703]
    training_std_x = [0.03555284748563298, 0.22732774983805104]
    x_s = x.copy()
    for i in range(2):
        x_s[:, :, i] = (x[:, :, i] - training_mean_x[i]) / training_std_x[i]

    predictions = pred_model.predict(x_s)
    predictions = predictions.squeeze()  # Now shape will be (24, 12)
    predictions = predictions.T

    training_mean_y = 23.022471007114433
    training_std_y = 2.917277113874284
    y_rescaled = (predictions * training_std_y) + training_mean_y

    F00 = np.zeros((L_gas, 24))
    for l in range(L_gas):
        for t in range(24):
            if l == 1 or l == 8:  # Skip pipeline if presence of a compressor
                continue
            node1 = pipes[l, 0]
            node2 = pipes[l, 1]
            if y_rescaled[node1 - 1, t] > y_rescaled[node2 - 1, t]:
                F00[l, t] = np.sqrt(y_rescaled[node1 - 1, t] ** 2 - y_rescaled[node2 - 1, t] ** 2)
            else:
                F00[l, t] = 1  # Replace NaN with 0 or another value ?

    # Set the parameters values
    GasDemandP.value = GasDemand[day]
    WindFactorP.value = WindFactor[day]
    F0.value = F00
    PR.value = y_rescaled

    # ----------------------------------------SOLVING OPTIMAIZATION PROBLEM -------------------------------------------

    result = prob.solve(solver='GUROBI', verbose=True, MIPGap=0.001)  # , MIPGap=0.0001 default value : 0.0001
    print(f"Day {day + 1} solved with status: {prob.status}")

    system_cost = elec_cost.value + gas_cost.value
    cost_tot.append(system_cost)

    P = cp.sum(p) * base
    print(f"GFPPs Power production: {P.value} [MWh]")
    G = cp.sum(g)
    print(f"Gas quantity bought: {G.value} [MNm3]")
    g_in_GFPP = cp.sum(GFPPmap @ p) * base / (ConversionFactor * Efficiency[0])
    print(f"Gas quantity in GFPPs: {g_in_GFPP.value} [MNm3]")
    print(f"Electricity system cost : {elec_cost.value} [€]")
    print(f"Gas system cost : {gas_cost.value} [€]")

    # -----------------------------------------------ANALYSES----------------------------------------------------------

    q_real = np.zeros((L_gas, T))
    pression = pr.value
    for t in range(T):
        for l in range(L_gas):
            node1 = pipes[l, 0]
            node2 = pipes[l, 1]
            if pression[node1 - 1, t] > pression[node2 - 1, t]:
                q_real[l, t] = K_w[l] * np.sqrt(pression[node1 - 1, t] ** 2 - pression[node2 - 1, t] ** 2)
            else:
                q_real[l, t] = 0

    mean_pipe_q = np.mean(q_real)
    q_diff = 100 * (q.value - q_real) / mean_pipe_q
    q_diff[1, :] = 0
    q_diff[8, :] = 0
    NMAE = np.mean(np.abs(q_diff))
    NMAE_tot.append(NMAE)

    q_MAE = np.abs(q.value - q_real)
    q_MAE[1, :] = 0
    q_MAE[8, :] = 0
    MAE = np.mean(q_MAE)
    MAE_tot.append(MAE)

    StandardDeviation = np.std(q_MAE)
    STD_tot.append(StandardDeviation)
    print('NMAE :', NMAE)
    print('MAE :', MAE)
    print('Standard deviation :', StandardDeviation)
    print_errors_table(q_diff)

    total_linepack = cp.sum(h, axis=0)
    total_initial_linepack = cp.sum(H_init, axis=0)
    linepack_variation = total_linepack.value - total_initial_linepack.value
    print_hourly_linepack(linepack_variation, T)

    GasProduction = cp.sum(g, axis=0).value
    GFPP_demand = cp.sum(GFPPmap @ p * base / (ConversionFactor * Efficiency[0]), axis=0).value
    #print_hourly_demand(GasDemand[day], GFPP_demand, GasProduction, T)

    ElecDemand = ElecDemand * base
    WindProd = Pwind[:, np.newaxis] * WindFactor[day]
    wind_prod = np.sum(WindProd * base, axis=0)
    GFPP_prod = cp.sum(p[-3:, :] * base, axis=0).value
    conv_prod = cp.sum(p[:2, :] * base, axis=0).value
    #print_hourly_demand_elec(ElecDemand, wind_prod, GFPP_prod, conv_prod, T)

    pressure = pr.value
    print_pressure_table(pressure, vmin=5, vmax=35)
    flow = q.value
    #print_flow_table(flow)

tac = time.perf_counter()
print('---GLOBAL RESULTS---')
print(f"Optimization problem solved in {tac - tic:0.4f} seconds")
print('Relative MAE [%]:', np.mean(NMAE_tot))
print('MAE [NMm3]:', np.mean(MAE_tot))
print('STD [NMm3]:', np.mean(STD_tot))
print('Daily cost [€]:', np.mean(cost_tot))


print('CHECK POINT : ** end of process **')
