import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def print_hourly_linepack(x, T):

    time_steps = range(T)  # Time steps (hours)

    time_steps = list(time_steps)

    # Plot the total linepack over time
    plt.plot(time_steps, x, marker='o')
    plt.title("Linepack variation compared to its initial value over 24 hours")
    plt.xlabel("Time (hours)")
    plt.ylabel("Total Linepack [MNm3]")
    plt.xlim(0, 24)
    plt.grid(True)
    plt.show()


def print_hourly_demand(x_demand, x_GFPP, x_production, T):

    time_steps = range(T)  # Time steps (hours)
    time_steps = list(time_steps)  # Add the 24th time step
    x_demand = x_demand + x_GFPP

    plt.plot(time_steps, x_demand, marker='o', label="Total Gas Demand", color='orange')  # Plot the gas demand over time
    plt.plot(time_steps, x_GFPP, marker='o', label="GFPP Demand", color='red')
    plt.plot(time_steps, x_production, marker='s', label="Gas Production", color='green', linestyle='--')  # Plot the gas production over time

    # Add plot title and labels
    plt.title("Gas Demand and Production Over 24 Hours")
    plt.xlabel("Time [hours]")
    plt.ylabel("Gas Volume [MNm3]")
    plt.xlim(0, 24)  # Set the x-axis limit to 0-24

    # Add grid, legend, and show plot
    plt.grid(True)
    plt.legend()  # Add legend to differentiate between demand and production
    plt.show()


def print_hourly_demand_elec(x_demand, x_wind, x_GFPP, x_conv, T):

    time_steps = range(T)  # Time steps (hours)
    time_steps = list(time_steps)  # Add the 24th time step

    plt.plot(time_steps, x_demand, marker='o', label="Electricity Demand", color='orange')  # Plot the gas demand over time
    plt.plot(time_steps, x_wind, marker='o', label="Wind farm production", color='green')
    plt.plot(time_steps, x_GFPP, marker='o', label="GFPP production", color='red')  # Plot the gas production over time
    plt.plot(time_steps, x_conv, marker='o', label="Nuclear production")

    # Add plot title and labels
    plt.title("Electricity Demand and Production Over 24 Hours")
    plt.xlabel("Time [hours]")
    plt.ylabel("Power [MW]")
    plt.xlim(0, 24)  # Set the x-axis limit to 0-24

    # Add grid, legend, and show plot
    plt.grid(True)
    plt.legend()  # Add legend to differentiate between demand and production
    plt.show()


def print_pressure_table(pressure_data, vmin, vmax):

    node_labels = [f" {i + 1}" for i in range(12)]  # Add 1 to each node number

    # Create the heatmap with Seaborn
    plt.figure(figsize=(8, 6))

    # Use Seaborn heatmap, with a red-blue colormap and custom vmin/vmax for the color scale
    cmap = sns.color_palette('copper_r', as_cmap=True)
    sns.heatmap(pressure_data, annot=True, cmap=cmap, linewidth=.3, yticklabels=node_labels, cbar_kws={'label': 'Pressure [bar]'},
                vmin=vmin, vmax=vmax)

    # Add labels and title
    plt.xticks(ticks=np.arange(0, 24, 4), labels=np.arange(0, 24, 4))
    plt.title("Pressure Evolution at Each Node Over Time", fontsize=16)
    plt.xlabel("Time (hours)", fontsize=14)
    plt.ylabel("Nodes", fontsize=14)

    # Show the plot
    plt.show()


def print_errors_table(error):

    node_labels = [f" {i + 1}" for i in range(12)]  # Add 1 to each node number

    # Create the heatmap with Seaborn
    plt.figure(figsize=(8, 6))

    # Use Seaborn heatmap, with a red-blue colormap and custom vmin/vmax for the color scale
    cmap = sns.color_palette('copper_r', as_cmap=True)
    ax = sns.heatmap(error, annot=False, vmin=-10, vmax=10, center=0, cmap='RdBu_r', linewidth=.3, yticklabels=node_labels, cbar_kws={'label': 'Error [%]'})

    # Add labels and title
    plt.xticks(ticks=np.arange(0, 24, 4), labels=np.arange(0, 24, 4))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    #plt.title("Gas Flow Error in Each Pipeline Over Time", fontsize=16)
    plt.xlabel("Time (hours)", fontsize=18)
    plt.ylabel("Pipeline", fontsize=18)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Error [%]', fontsize=16)

    # Show the plot
    plt.show()

def print_flow_table(flow):

    node_labels = [f" {i + 1}" for i in range(12)]  # Add 1 to each node number

    # Create the heatmap with Seaborn
    plt.figure(figsize=(8, 6))

    # Use Seaborn heatmap, with a red-blue colormap and custom vmin/vmax for the color scale
    sns.heatmap(flow, annot=False, center=0, cmap='RdBu_r', linewidth=.3, yticklabels=node_labels,
                cbar_kws={'label': 'Flow [MNm3]'})

    # Add labels and title
    plt.xticks(ticks=np.arange(0, 24, 4), labels=np.arange(0, 24, 4))
    plt.title("Gas Flow in Each Pipeline Over Time", fontsize=16)
    plt.xlabel("Time (hours)", fontsize=14)
    plt.ylabel("Pipeline", fontsize=14)

    # Show the plot
    plt.show()

