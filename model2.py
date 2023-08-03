import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class Conv_QNet(nn.Module):
    def __init__(self, input_shape, hidden_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1) # 3 input channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.fc1_size = self._get_fc1_size(input_shape)
        self.fc1 = nn.Linear(self.fc1_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def _get_fc1_size(self, input_shape):
        # Dummy forward pass to calculate flattened size
        x = torch.zeros(1, *input_shape) # Only the channels, height, and width
        #print(x.shape)
        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = F.relu(self.conv2(x))
        #print(x.shape)
        #print(f'get_fc1_size:{x.view(x.size(0), -1).shape}')
        flattened_size = x.view(x.size(0), -1).size(1)
        return flattened_size # Just return the flattened size for a single example




    def forward(self, x):
        #print(f"Input Shape: {x.shape}")  # Debugging print
        # A terrible way to do this but I am going to add in the batch dimension here. This is mostly to see if this fixes my bug with the dim mismatch
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #print("After Convolution Shape:", x.shape)  # Debugging print
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        #print("Flattened Shape:", x.shape)  # Debugging print
        #print("FC1 Size:", self.fc1_size)  # Debugging print
        x = F.relu(self.fc1(x)) # This line is where the error is raised
        x = self.fc2(x)
        return x


    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float) # add channel dimension if not present
        state = torch.squeeze(state)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 3:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
        
        #Add another dimension for the batch
        if len(action.shape) == 1:
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        #print(f'State shape before pred: {state.shape}')
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()