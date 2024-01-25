import torch
import pandas as pd
from torch_geometric.data import HeteroData
from ast import literal_eval

def main():
    experts_df = pd.read_csv("../data/Users.csv", converters={"skills":literal_eval})
    teams_df = pd.read_csv("../data/Teams.csv", converters={"required_skills":literal_eval, "members":literal_eval})

    experts_df['skillset'] = [str(row) for row in experts_df['skills']]
    teams_df['required_skillset'] = [str(row) for row in teams_df["required_skills"]]
    all_skills = set(skill for skills_list in experts_df['skills'] for skill in skills_list)
    skills_df = pd.DataFrame(list(enumerate(all_skills, start=1)), columns=["skill_id", "skill_name"])
    skill_mapping = {skill_name: skill_id for skill_id, skill_name in zip(skills_df['skill_id'], skills_df['skill_name'])}
    experts_df['skills'] = experts_df['skills'].apply(lambda skills_list: [skill_mapping[skill] for skill in skills_list])
    teams_df['required_skillset'] = teams_df['required_skills'].apply(lambda skills_list: [skill_mapping[skill] for skill in skills_list])

    # Create edge index for expert-skill
    edges_expert_skill = []
    for _, row in experts_df.iterrows():
        expert_index = row['user_id']
        for skillID in row['skills']:
            skill_index = skillID
            edges_expert_skill.append((expert_index, skill_index))

    edge_index_expert_skill = torch.tensor(list(zip(*edges_expert_skill)), dtype=torch.long)

    # Create edge index for team-skill
    edges_team_skill = []
    for _, row in teams_df.iterrows():
        team_index = row['team_id']
        for skillID in row['required_skillset']:
            skill_index = skillID
            edges_team_skill.append((team_index, skill_index))

    edge_index_team_skill = torch.tensor(list(zip(*edges_team_skill)), dtype=torch.long)

    # Create edge index for team-experts
    edges_team_experts = []
    for _, row in teams_df.iterrows():
        team_index = row['team_id']
        for expertID in row['members']:
            expert_index = expertID
            edges_team_experts.append((team_index, expert_index))

    edge_index_team_experts = torch.tensor(list(zip(*edges_team_experts)), dtype=torch.long)

    data = HeteroData()

    data['expert'].x = torch.tensor(experts_df["user_id"].values, dtype=torch.long)
    data['skill'].x = torch.tensor(skills_df["skill_id"].values, dtype=torch.long)
    data['team'].x = torch.tensor(teams_df["team_id"].values, dtype=torch.long)


    data['team', 'requires', 'skill'].edge_index = edge_index_team_skill
    data['team', 'includes', 'expert'].edge_index = edge_index_team_experts
    data['expert', 'has', 'skill'].edge_index = edge_index_expert_skill

    data['team', 'requires', 'skill'].edge_attr = None
    data['team', 'includes', 'expert'].edge_attr = None
    data['expert', 'has', 'skill'].edge_attr = None

    import torch_geometric.transforms as T
    data = T.ToUndirected()(data)
    torch.save(data, 'data.pt')
    return data
