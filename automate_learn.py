import os


def main():
    num_experiments = 1
    start_in = 0

    for i in range(start_in, start_in + num_experiments):
        os.system("mlagents-learn config/trainer_config.yaml --env=./unity-volume/windows/bomberman/Bomberman.exe --train --run-id=ppo_first_test_" + str(i))

if __name__ == '__main__':
    main()