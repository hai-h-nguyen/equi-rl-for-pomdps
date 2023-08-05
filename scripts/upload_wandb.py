import wandb
import csv
import argparse
import glob
import time

parser = argparse.ArgumentParser()

parser.add_argument('--folder', type=str)

args = parser.parse_args()

csv_files = []
for file in glob.glob(args.folder + '/*.csv'):
    print(file)
    splitted = file.split('/')
    group = splitted[1]
    project = splitted[2]
    name = splitted[3][:-4]

    wandb.init(project=project,
               settings=wandb.Settings(_disable_stats=True),
               entity='hainh22',
               group=group,
               name=name)

    with open(file, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        i = 0

        for row in csv_reader:
            if i > 0:
                wandb.log({'Reward': float(row[2])}, step=int(row[1]))
                time.sleep(0.01)
            i += 1

    wandb.finish()