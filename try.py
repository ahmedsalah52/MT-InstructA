import threading
from  tqdm import tqdm
def get_episode(num,num2,ret):
    ret.append(num+num2)


episodes = 10
data_dict = {}
taskname = 'button-press-topdown-v2'
threads = []
rets = []
args = [1,2,rets]
for i in range(episodes):
    t = threading.Thread(target=get_episode, args=(args))  
    t.start()
    threads.append(t)

for t in tqdm(threads):
    t.join()
    
for episode_num in range(episodes):
    episode_dict = rets[episode_num]
    data_dict[episode_num] = {'task_name':taskname,'data':episode_dict}

print(data_dict)




class CustomMLP(BaseFeaturesExtractor):


    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, net_arch = [256]):
        super().__init__( observation_space = observation_space,features_dim = net_arch[-1])
        n_input_channels = observation_space.shape[0]

        net_arch = net_arch[:-1] # exclude the last layer, it will be added in the 
        modules = []
        for i , layer_dim in enumerate(net_arch):
            modules.append(nn.Linear(n_input_channels, layer_dim))
            if i == 0:
                modules.append(nn.LayerNorm(layer_dim))
            else:
                modules.append(LeakyReLU())
            n_input_channels = layer_dim

        self.linear = nn.Sequential(*modules)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        
        features = self.linear(observations)


        return features

