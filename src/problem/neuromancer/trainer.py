"""
Training pipeline
"""

from tqdm import tqdm
import torch

def train(components, loss_fn, loader_train, loader_dev, loader_test,
          optimizer, epochs, patience, warmup, loss_key="loss"):
    """
    training neuromancer with penalized loss
    """
    # define counter for early stopping
    early_stop_counter = 0
    # calculate init validation loss
    val_loss = 0.0
    with torch.no_grad():
        for data_dict in loader_dev:
            data_dict.update(components(data_dict))
            val_loss += loss_fn(data_dict)[loss_key].item()
    val_loss /= len(loader_dev)
    best_loss = val_loss
    # training
    for epoch in tqdm(range(epochs)):
        tqdm.write("Epoch {}, Validation Loss: {:.2f}".format(epoch, val_loss))
        # set as train mode
        components.train()
        # iterate through training data
        for data_dict in loader_train:
            # forward pass
            data_dict.update(components(data_dict))
            data_dict = loss_fn(data_dict)
            # backward pass
            data_dict[loss_key].backward()
            # update parameters
            optimizer.step()
            # clear grads
            optimizer.zero_grad()
        # validation after every epoch
        components.eval()
        # calculate validation loss
        val_loss = 0.0
        with torch.no_grad():
            for data_dict in loader_dev:
                data_dict.update(components(data_dict))
                val_loss += loss_fn(data_dict)[loss_key].item()
        val_loss /= len(loader_dev)
        # check if better
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = components.state_dict()  # store best parameters
            early_stop_counter = 0  # reset counter
        else:
            early_stop_counter += 1
        # early stop
        if epoch >= warmup and early_stop_counter >= patience:
            tqdm.write("Early stopping at epoch {}".format(epoch))
            break
    # end of epochs
    tqdm.write("Epoch {}, Validation Loss: {:.2f}".format(epoch, val_loss))
    # load best model
    components.load_state_dict(best_model_state)
    print("Training complete.")
