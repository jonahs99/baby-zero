import torch
from torch.autograd import Variable
import torch.utils.data as data_utils

import matplotlib.pyplot as plt

class ModelInterface:
    def __init__(self, model):
        self.model = model
        self.load_path = ''
        self.save_path = ''

        self.train_data = {'inputs': [], 'nodes': []}
        self.loss_history = []

    def load_parameters(self, name):
        import os
        indices = [int(f[:-3].split('_')[1]) for f in os.listdir('./data/') if f.startswith(name)]
        if len(indices) > 0:
            self.load_path = './data/' + name + '_' + str(max(indices)) + '.pt'
            self.save_path = './data/' + name + '_' + str(max(indices) + 1) + '.pt'
            self.model.net.load_state_dict(torch.load(self.load_path))
        else:
            self.load_path = ''
            self.save_path = './data/' + name + '_0.pt'
    
    def save_parameters(self):
        torch.save(self.model.net.state_dict(), self.save_path)

    def predict(self, state, node):
        inputs = self.model.represent(state)
        self.train_data['inputs'].append(inputs)
        self.train_data['nodes'].append(node)
        return self.model.net(Variable(inputs)).data[0, 0]

    def clear_training(self):
        self.train_data = {'inputs': [], 'nodes': []}

    def train(self):
        BATCH_SIZE = 32
        EPOCHS = 1
        visit_threshold = 10

        n_data = len(self.train_data['inputs'])

        print('Collecting', n_data, 'training data...')

        input_tensor = torch.FloatTensor(n_data, *self.model.INPUT_SIZE)
        target_tensor = torch.FloatTensor(n_data, 1)

        i = 0
        for inputs, node in zip(self.train_data['inputs'], self.train_data['nodes']):
            if node.n >= visit_threshold:
                input_tensor[i] = inputs
                target_tensor[i, 0] = node.w / node.n
                i += 1
        
        print('Kept', i, 'data.')

        input_tensor = input_tensor[0:i]
        target_tensor = target_tensor[0:i]

        data_set = data_utils.TensorDataset(input_tensor, target_tensor)
        data_loader = data_utils.DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True)

        print('Beginning', EPOCHS, 'epochs...')

        for epoch in range(EPOCHS):
            epoch_loss = 0
            batch_count = 0
            for data in data_loader:
                self.model.optimizer.zero_grad()
                output = self.model.net(Variable(data[0]))
                loss = self.model.criterion(output, Variable(data[1]))
                epoch_loss += loss.data[0]
                loss.backward()
                self.model.optimizer.step()

                batch_count += 1
            self.loss_history.append(epoch_loss / batch_count)
            if epoch % max(1, EPOCHS // 10) == 0:
                print(epoch)
    
    def plot_loss(self):
        plt.plot(self.loss_history)
        plt.xlabel('epoch')
        plt.ylabel('average batch mse')
        plt.draw()
        plt.pause(0.001)