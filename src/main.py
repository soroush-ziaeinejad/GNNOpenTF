import dataPreparation
import model
import glob
from ast import literal_eval
import pandas as pd
import torch

files_t = sorted(glob.glob('../data/new_data/*/Teams.csv'))
files_u = sorted(glob.glob('../data/new_data/*/Users.csv'))
for i in range(len(files_t)):
    if files_t[i].split('\\')[1] not in ['gith']: continue
    print(files_t[i], files_u[i])
    experts_df = pd.read_csv(files_u[i], converters={"skills": literal_eval})
    teams_df = pd.read_csv(files_t[i], converters={"required_skills": literal_eval, "members": literal_eval})
    try:
        data = torch.load(files_t[i].split('\\')[0] + '/' + files_t[i].split('\\')[1] + '/data.pt')
        print('data loaded')
    except:
        data = dataPreparation.main(experts_df, teams_df, files_t[i].split('\\')[0] + '/' + files_t[i].split('\\')[1] + '/data.pt')
        print('data saved')
    final_model = model.main(data, files_t[i].split('\\')[1])
    torch.save(final_model, '../output/NewSplitMethod' + '/' + files_t[i].split('\\')[1] + '/model.pt')
    # torch.save(final_model, files_t[i].split('\\')[0] + '/' + files_t[i].split('\\')[1] + '/model.pt')
