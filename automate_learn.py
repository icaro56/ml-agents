import os


def main():
    num_experiments = 1
    start_in = 1

    for i in range(start_in, start_in + num_experiments):
        # os.system("mlagents-learn config/trainer_config_bc.yaml --env=./unity-volume/windows/bomberman_1/Bomberman.exe --train --load --run-id=bc_ppo" + str(i))
        # os.system("mlagents-learn config/trainer_config_bc.yaml --env=./unity-volume/windows/bomberman_2/Bomberman.exe --train --run-id=bc_" + str(i))
        os.system("mlagents-learn config/trainer_config_bc.yaml --env=./unity-volume/windows/bomberman_student/Bomberman.exe --train --load --run-id=ppo_four_" + str(i))
        # os.system("mlagents-learn config/trainer_config.yaml --env=./unity-volume/windows/bomberman/Bomberman.exe --train --load --no-graphics --run-id=hybrid_one_random_scenario_" + str(i))


if __name__ == '__main__':
    main()