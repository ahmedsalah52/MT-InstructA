import pickle
solution = 'buffers/sac_replay_buffer_0.pkl'
with open(solution, 'rb') as handle:
    solution_dict = pickle.load(handle)
print(solution_dict)
for k,v in solution_dict:
    print(k)