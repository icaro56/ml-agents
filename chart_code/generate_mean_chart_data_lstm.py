import json
import pandas as pd
import math

files_num = 5
statistic_num = 6
representation_num = 5
foldername = "../chart_data/bomb/"

json_datas = []
json_mean_data = []

def convertIntToStatistic(value):
    if value == 1: return "cumulative_reward"
    elif value == 2: return "entropy"
    elif value == 3: return "episode_length"
    elif value == 4: return "policy_loss"
    elif value == 5: return "value_estimate"
    elif value == 6: return "value_loss"
    else: return "null"

def convertIntToRepresentation(value):
    if value == 6:
        return "hybrid_lstm"
    else:
        return "null"

def getFinalFilename(statisticName, representationState, sufix_num):
    return foldername + representationState + "/" + statisticName + "/run_bomberman_agents_" + representationState + "_" + str(sufix_num) + "-0-tag-Info_" + statisticName + ".json"

representation_index = 6
REPRESENTATION_STATE = convertIntToRepresentation(representation_index)

for statistic_index in range(1, statistic_num + 1):
    STATISTIC_NAME = convertIntToStatistic(statistic_index)

    for sufix_num in range(1, files_num + 1):
        with open(getFinalFilename(STATISTIC_NAME, REPRESENTATION_STATE, sufix_num)) as datafile:
            data = json.load(datafile)
            json_datas.append(data)

    maxIteration = 0
    maxIndex = 0
    for m in range(0, files_num):
        if (len(json_datas[m]) > maxIteration):
            maxIteration = len(json_datas[m])
            maxIndex = m

    # calculando a média
    for i in range(0, len(json_datas[m])):
        mean_value = 0
        for j in range(0, files_num):
            if (i < len(json_datas[j])):
                mean_value = mean_value + json_datas[j][i][2]
            else:
                mean_value = mean_value + json_datas[j][len(json_datas[j]) - 1][2]

        mean_value = mean_value / files_num

        line = [0, json_datas[m][i][1], mean_value]
        json_mean_data.append(line)

    print(json_mean_data)

    # calculando o desvio padrão
    for i in range(0, len(json_datas[m])):
        mean_value = json_mean_data[i][2];
        variance = 0

        for j in range(0, files_num):
            if (i < len(json_datas[j])):
                variance = variance + ((json_datas[j][i][2] - mean_value) ** 2)
            else:
                variance = variance + ((json_datas[j][len(json_datas[j]) - 1][2] - mean_value) ** 2)

        variance = variance / (files_num-1)
        std_deviation = math.sqrt(variance)

        json_mean_data[i][0] = std_deviation

    print(json_mean_data)

    with open(foldername + REPRESENTATION_STATE + "/" + STATISTIC_NAME + "/mean.json", "w") as outfile:
        json.dump(json_mean_data, outfile)

    # limpando lista
    json_datas.clear()
    json_mean_data.clear()