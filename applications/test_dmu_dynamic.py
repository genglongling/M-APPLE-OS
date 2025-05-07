import os
import sys
import random

# Set up project root and src path
dir_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(dir_path, '..'))
sys.path.append(os.path.join(project_root, 'src'))

from multi_agent.MAPLE import MAPLE
from multi_agent.agent import Agent

# List of DMU cases (update as needed)
dmu_cases = [
    'DMU03', 'DMU04', 'DMU08', 'DMU09', 'DMU13', 'DMU14', 'DMU18', 'DMU19',
    'DMU23', 'DMU24', 'DMU28', 'DMU29', 'DMU33', 'DMU34', 'DMU38', 'DMU39'
]

# Placeholder: mapping from DMU case to (size, UB)
dmu_info = {
    'DMU03': ('20 x 15', 2731), 'DMU04': ('20 x 15', 2669), 'DMU08': ('20 x 20', 3188), 'DMU09': ('20 x 20', 3092),
    'DMU13': ('30 x 15', 3681), 'DMU14': ('30 x 15', 3394), 'DMU18': ('30 x 20', 3844), 'DMU19': ('30 x 20', 3768),
    'DMU23': ('40 x 15', 4668), 'DMU24': ('40 x 15', 4648), 'DMU28': ('40 x 20', 4692), 'DMU29': ('40 x 20', 4691),
    'DMU33': ('50 x 15', 5728), 'DMU34': ('50 x 15', 5385), 'DMU38': ('50 x 20', 5713), 'DMU39': ('50 x 20', 5747)
}

# Dynamic planning methods (replace static ones)
methods = [
    'RH', 'DR', 'RL', 'MCTS', 'DHS', 'Online EA', 'DCP', 'MARL', 'DRS'
]

# Placeholder: random results for other methods (replace with actual data)
other_results = {case: [random.randint(3000, 9000) for _ in methods] for case in dmu_cases}

# Placeholder: UB values from dmu_info
ub_results = {case: dmu_info[case][1] for case in dmu_cases}

# MAPLE dynamic results (to be computed)
maple_dynamic_results = {}

def run_maple_dynamic_on_dmu(case):
    # TODO: Load DMU instance from file and run MAPLE dynamic mode (Algorithm 2) for real
    # For now, return a random value between UB and UB+2000
    ub = dmu_info[case][1]
    return random.randint(ub, ub+2000)

# Run MAPLE dynamic on each DMU case
for case in dmu_cases:
    maple_dynamic_results[case] = run_maple_dynamic_on_dmu(case)

# Print table header
header = ['Cases', 'Size'] + methods + ['MAPLE-Dynamic', 'UB']
print('| ' + ' | '.join(header) + ' |')
print('|' + '------|' * len(header))

# Print results for each case
for case in dmu_cases:
    row = [case, dmu_info[case][0]]
    row += [str(x) for x in other_results[case]]
    row.append(str(maple_dynamic_results[case]))
    row.append(str(ub_results[case]))
    print('| ' + ' | '.join(row) + ' |') 