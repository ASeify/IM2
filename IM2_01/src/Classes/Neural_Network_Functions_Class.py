# Version 0.0.1

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import sys
import os
from tqdm import tqdm
from datetime import datetime
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if not (CURRENT_DIR in sys.path):
    sys.path.append(CURRENT_DIR)

from Bcolors_Class import Bcolors as bcolors
from Average_Meter_Class import AverageMeter
from Neural_Network_Models_Class import Multilayer_Full_Model
from Neural_Network_Models_Class import Node_Embedding_Model
from Neural_Network_Models_Class import Node_Layer_Embedding_Model
from Files_Handler_Class import Files_Handler

file_handler_obj = Files_Handler()

class Neural_Network_Functions:

    @staticmethod
    def get_device():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            torch.cuda.empty_cache()
        return device

    @staticmethod
    def train_one_epoch(model, train_loader:DataLoader,
                    loss_fn:nn.modules.activation, optimizer:torch.optim,
                    epoch:int=None, device:str='cuda'):
        model.train()
        loss_train = AverageMeter()
        epoch_loss_hostory = []
        with tqdm(train_loader, unit=" batch") as tepoch:
            for node_inputs, layer_inputs, targets in tepoch:
                if epoch is not None:
                    tepoch.set_description(f"Train Epoch {epoch + 1}")
                node_inputs = node_inputs.to(device)
                layer_inputs = layer_inputs.to(device)
                # targets = targets.to(device)

                outputs = model(node_inputs, layer_inputs).squeeze(dim=1)
                loss = loss_fn(outputs.to(device), targets.to(device))
                # print(loss)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                optimizer.step()
                optimizer.zero_grad()

                epoch_loss_hostory.append(loss.item())
                loss_train.update(loss.item())
                tepoch.set_postfix(loss=loss_train.avg)
            return model, loss_train.avg, epoch_loss_hostory

    @staticmethod
    def validation(model, test_loader:DataLoader,
               loss_fn:nn.modules.activation, epoch:int=None, device:str='cuda'):
        model.eval()
        epoch_loss_hostory = []
        with tqdm(test_loader, unit=" batch") as tepoch:
            with torch.no_grad():
                loss_valid = AverageMeter()
                # acc_valid = Accuracy().to(device)
                for node_inputs, layer_inputs, targets in tepoch:
                    if epoch is not None:
                        tepoch.set_description(f"Test  Epoch {epoch + 1}")
                        node_inputs = node_inputs.to(device)
                        layer_inputs = layer_inputs.to(device)
                        # targets = targets.to(device)

                        outputs = model(node_inputs, layer_inputs).squeeze(dim=1)
                        loss = loss_fn(outputs.to(device), targets.to(device))

                        epoch_loss_hostory.append(loss.item())
                        loss_valid.update(loss.item())
                        tepoch.set_postfix(loss=loss_valid.avg)
                        # acc_valid(outputs, targets.int())
        return loss_valid.avg, epoch_loss_hostory

    @staticmethod
    def create_Multilayer_Full_Model_model (node_in_features:int=10, node_out_features:int=256,
                                            layer_in_features:int=6, layer_out_features:int=256,
                                            encoder_head:int=32, num_encoder:int=2,
                                            bias:bool=True, dropout:float=0.05, device:str='cpu'):
        model = Multilayer_Full_Model(
                        node_in_features=node_in_features, node_out_features=node_out_features,
                        layer_in_features=layer_in_features, layer_out_features=layer_out_features,
                        encoder_head=encoder_head, num_encoder=num_encoder, encoder_activation='gelu',
                        bias=bias, dropout=dropout,
                        activation=nn.GELU(), device=device).to(device)
        return model

    @staticmethod
    def create_Node_Layer_Embedding_Model (node_in_features:int=10, node_out_features:int=256,
                                            layer_in_features:int=6, layer_out_features:int=256,
                                            bias:bool=True, device:str='cpu'):
        model = Node_Layer_Embedding_Model(
                        node_in_features=node_in_features, node_out_features=node_out_features,
                        layer_in_features=layer_in_features, layer_out_features=layer_out_features,
                        bias=bias, activation=nn.GELU(), device=device).to(device)
        return model

    @staticmethod
    def create_Node_Embedding_Model (node_in_features:int=10, node_out_features:int=256,
                                            bias:bool=True, device:str='cpu'):
        model = Node_Embedding_Model(
                        node_in_features=node_in_features, node_out_features=node_out_features,
                        bias=bias, activation=nn.GELU(), device=device).to(device)
        return model

    @staticmethod
    def small_gride(create_model, train_loader:DataLoader, loss_fn:nn.HuberLoss, epoch_cun:int, device:str='cpu'):
        best_lr = 0.0001
        best_wd = 1e-5
        delta = -1
        num_epochs = epoch_cun
        for lr in [0.01, 0.009, 0.007, 0.005, 0.003, 0.001, 0.0009, 0.0007, 0.0005, 0.0003, 0.0001]:
            for wd in [1e-4, 1e-5, 0.]:
                model = create_model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
                print(f'LR={lr}, WD={wd}')
                start_loss = torch.inf
                end_loss = torch.inf
                for epoch in range(num_epochs):
                    model, loss, _ = Neural_Network_Functions.train_one_epoch(model, train_loader, loss_fn, optimizer, epoch, device)
                if epoch == 0:
                    start_loss = loss
                else:
                    end_loss = loss
                if (start_loss - end_loss) > delta:
                    delta = start_loss - end_loss
                    best_lr = lr
                    best_wd = wd
        return best_lr, best_wd

    @staticmethod
    def get_load_model_info(loaded_model:str, best_train_model:bool, manual_select_model:bool=False):
        load_model_status = False
        loaded_model_info = None
        if manual_select_model:
            loaded_model = file_handler_obj.select_files("Model file", ".pt", False)
        loaded_lr, loaded_wd, loaded_epochs = None, None, None
        if loaded_model != '':
            loaded_model_info = file_handler_obj.get_file_path_info(loaded_model)
            if best_train_model:
                loaded_lr = float(loaded_model_info['name'].split(" ")[2].split("=")[1])
                loaded_wd = float(loaded_model_info['name'].split(" ")[3].split("=")[1])
                loaded_epochs = int(loaded_model_info['name'].split(" ")[4].split("=")[1])
            else:
                loaded_lr = float(loaded_model_info['name'].split(" ")[2].split("=")[1])
                loaded_wd = float(loaded_model_info['name'].split(" ")[3].split("=")[1])
                loaded_epochs = int(loaded_model_info['name'].split(" ")[4].split("=")[1])
            load_model_status = True
        if load_model_status:
            print(bcolors.OKGREEN + f"Load model: {load_model_status}" + bcolors.ENDC)
            print()
            print(bcolors.OKBLUE + loaded_model_info['path']+ bcolors.ENDC)
            print(bcolors.WARNING + loaded_model_info['name']+ bcolors.ENDC)
            print()
            print(bcolors.OKGREEN + f"loaded_lr: {loaded_lr}"  + bcolors.ENDC)
            print(f"loaded_wd: {loaded_wd}")
            print(bcolors.FAIL + f"loaded_epochs: {loaded_epochs}" + bcolors.ENDC)
        else:
            print(bcolors.FAIL + f"Load model: {load_model_status}" + bcolors.ENDC)

        return load_model_status, loaded_model_info, loaded_lr, loaded_wd, loaded_epochs
    
    @staticmethod
    def num_params(model, scale:int=1000):
        nums = sum(p.numel() for p in model.parameters()) / scale
        return nums
    
    @staticmethod
    def get_loaded_optimizer(loaded_optimizer, model, lr, wd, device):
        if loaded_optimizer != '':
            optimizer = torch.load(loaded_optimizer, map_location=torch.device(device))
            print(bcolors.OKGREEN + 'Optimizer load' + bcolors.ENDC)
        else:
            optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=wd)
            print(bcolors.FAIL + 'Optimizer create' + bcolors.ENDC)

        lr = optimizer.param_groups[0]['lr']
        wd = optimizer.param_groups[0]['weight_decay']
        print("lr = " + str(lr), "\nwd = " + str(wd))
        return optimizer

    @staticmethod
    def create_result_dir(load_model_status, loaded_model_info, dir_name, model_info):
        current_date = datetime.now()
        model_date = (str(current_date.year) + "_" + str(current_date.month) + "_" +
                    str(current_date.day) + "_" + str(current_date.hour) + "_" +
                    str(current_date.minute))
        if load_model_status:
            source_code_path = loaded_model_info['path'][:-1][:loaded_model_info['path'][:].rfind("/")]
        else:
            source_code_path = file_handler_obj.make_dir(str(os.getcwd()), f'/{dir_name}')
        source_code_path = source_code_path.replace("\\", "/")
        print(source_code_path)
        source_code_path = file_handler_obj.make_dir(source_code_path, str('/' + str(model_date)) + ' ' + model_info)
        source_code_path = source_code_path.replace("\\", "/")
        print(source_code_path)
        return source_code_path

    @staticmethod
    def plot_train_progress(epoch_counter, loss_train_hist, loss_valid_hist, save_path):
        plt.plot(range(epoch_counter), loss_train_hist, "r-", label="Train")
        plt.plot(range(epoch_counter), loss_valid_hist, "b-", label="Validation")

        plt.xlabel("Epoch: " + str(epoch_counter))
        plt.ylabel("loss: "
                + "T=" + str(f"{loss_train_hist[-1]:.4}")
                + " & "
                + "V=" + str(f"{loss_valid_hist[-1]:.4}")
        )
        x_spacing = 25
        y_spacing = 5
        x_minorLocator = MultipleLocator(x_spacing)
        y_minorLocator = MultipleLocator(y_spacing)
        plt.grid(visible=True, alpha=0.8, linewidth=1)
        plt.legend()
        ax = plt.gca()
        ax.yaxis.label.set_fontsize('large')
        ax.xaxis.label.set_fontsize('large')
        ax.yaxis.set_minor_locator(y_minorLocator)
        ax.xaxis.set_minor_locator(x_minorLocator)
        ax.grid(which = 'minor')
        plt.savefig(
            save_path
            + "epoch=" + str(len(loss_valid_hist))
            + " loss_valid=" + str(f"{loss_valid_hist[-1]:.5}")
            + " loss_train=" + str(f"{loss_train_hist[-1]:.5}")
            + ".png"
        )
        pass

    pass