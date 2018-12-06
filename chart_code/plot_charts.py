import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from scipy.interpolate import interp1d

chart_data = []
dataframes = []
plots = []

step_error_num = 50
files_num = 5
statistic_num = 6
representation_num = 5
statistic_value = 6

def convertIntToStatistic(value):
    if value == 1: return "cumulative_reward"
    elif value == 2: return "entropy"
    elif value == 3: return "episode_length"
    elif value == 4: return "policy_loss"
    elif value == 5: return "value_estimate"
    elif value == 6: return "value_loss"
    else: return "null"

def convertIntToStatisticView(value):
    if value == 1: return "Recompensa Acumulada"
    elif value == 2: return "Entropia"
    elif value == 3: return "Tamanho do Episódio"
    elif value == 4: return "Loss da Política"
    elif value == 5: return "Estimativa de Valor"
    elif value == 6: return "Loss do Valor"
    else: return "null"

def convertIntToRepresentation(value):
    if value == 1: return "binary_flag"
    elif value == 2: return "binary_normalized"
    elif value == 3: return "hybrid"
    elif value == 4: return "icaart"
    elif value == 5: return "zero_or_one"
    else: return "null"

def convertIntToRepresentationLegends(value):
    if value == 1: return "Flag Binária"
    elif value == 2: return "Flag Binária Norm."
    elif value == 3: return "Híbrida"
    elif value == 4: return "ICAART"
    elif value == 5: return "Zero ou Um"
    else: return "null"

def convertRepresentationIntToColor(value):
    if value == 1: return 'gold'
    elif value == 2: return 'm'
    elif value == 3: return 'r'
    elif value == 4: return 'blue'
    elif value == 5: return 'yellowgreen'
    else: return 'k'



STATISTIC_NAME = convertIntToStatistic(statistic_value)
STATISTIC_NAME_TO_VIEW = convertIntToStatisticView(statistic_value)

for representation_index in range(2, representation_num+1):
    REPRESENTATION_STATE = convertIntToRepresentation(representation_index)
    with open("../chart_data/bomb_multi_brain/" + REPRESENTATION_STATE + "/" + STATISTIC_NAME + "/mean.json") as datafile:
        temp = json.load(datafile)
        # temp.insert(0, [0.0, 0, 0.0])
        chart_data.append(temp)

for i, value in enumerate(chart_data):
    df = pd.DataFrame(value);
    dataframes.append(df)

fig, ax = plt.subplots()

mkfunc = lambda x, pos: '%1.0fM' % (x * 1e-6) if x >= 1e6 else '%1.0fK' % (x * 1e-3) if x >= 1e3 else '%1.0f' % x
formatter = FuncFormatter(mkfunc)
ax.xaxis.set_major_formatter(formatter)

for index, value in enumerate(dataframes):
    x_values = value.iloc[:, 1]
    y_values = value.iloc[:, 2]

    # fiz uma especie de linspace com o dataframe
    # x_smooth = np.linspace(x_values.min(), x_values.max(), 50)

    x_dataframe_1 = value.iloc[::step_error_num, 1].append(value.iloc[[-1], 1])
    # y_dataframe_1 = value.iloc[::step_error_num, 2].append(value.iloc[[-1], 2])
    error_dataframe_1 = value.iloc[::step_error_num, 0].append(value.iloc[[-1], 0])

    f2 = interp1d(x_values, y_values, kind='cubic', bounds_error=False)

    p, = ax.plot(x_dataframe_1,
                 f2(x_dataframe_1),
                 label=convertIntToRepresentation(index + 1),
                 color=convertRepresentationIntToColor(index + 1),
                 alpha=0.8)

    plots.append(p)

    ax.errorbar(x_dataframe_1,
                f2(x_dataframe_1),
                yerr=error_dataframe_1,
                color=convertRepresentationIntToColor(index + 1),
                label="error_" + convertIntToRepresentation(index + 1),
                fmt="none",
                alpha=0.5,
                elinewidth=2.0,
                capsize=3,
                capthick=3)

plots_names = [convertIntToRepresentationLegends(i + 2) for i, x in enumerate(plots)]
ax.legend(plots,
          plots_names,
          loc=0,
          prop={'size': 12},
          fancybox=True, framealpha=1.0)

# ax.set(title='Comparação de ' + STATISTIC_NAME_TO_VIEW)

# fig.suptitle('Comparação de ' + STATISTIC_NAME_TO_VIEW, fontsize=18)
ax.set_xlabel('Número de Iterações', fontsize=16)
ax.set_ylabel(STATISTIC_NAME_TO_VIEW, fontsize=16)

ax.grid()
plt.xticks(x_dataframe_1)

def on_resize(event):
    fig.tight_layout()
    fig.canvas.draw()

cid = fig.canvas.mpl_connect('resize_event', on_resize)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.show()
