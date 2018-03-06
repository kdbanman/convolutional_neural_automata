import torch

class AutomataConvolutionTrainer:

    def train_by_iteration(self, conv_net, reference_environment, iterations_to_train = 5000, learning_rate = 1e-2, progress_callback = None):
        loss_fn = torch.nn.MSELoss(size_average=False)
        optimizer = torch.optim.Adam(conv_net.parameters(), lr=learning_rate)

        loss_history = []
        previous_callback_percent = 0
        for iteration in range(0, iterations_to_train):
            reference_environment.iterate()

            input_grid = torch.autograd.Variable(torch.from_numpy(reference_environment.previous_grid))
            known_output = torch.autograd.Variable(torch.from_numpy(reference_environment.current_grid))
            predicted_output = conv_net(input_grid)
            
            loss = loss_fn(predicted_output, known_output)

            loss_history.append(loss.data[0])
            progress_percent = iteration // iterations_to_train
            if progress_callback is not None and progress_percent > previous_callback_percent:
                progress_callback(iteration, progress_percent, loss.data[0])
                previous_callback_percent = progress_percent

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        
        return loss_history