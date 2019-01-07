import os


def main():
    num_experiments = 1
    start_in = 15

    for i in range(start_in, start_in + num_experiments):
        os.system("mlagents-learn config/trainer_config.yaml --env=./unity-volume/windows/bomberman/Bomberman.exe --train --load --no-graphics --run-id=calibrando_rewards_" + str(i))
        # os.system('mlagents-learn config/trainer_config.yaml --env="./unity-volume/windows/hallway/Unity Environment.exe" --train --run-id=hallway4_' + str(i))

if __name__ == '__main__':
    main()