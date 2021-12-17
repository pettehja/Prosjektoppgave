from os import path, mkdir
import torch.utils.data
import matplotlib.pyplot as plt
import pandas as pd
import time
from neuralNet import NeuralNetwork
from train import train

rseed = 12345
torch.manual_seed(rseed)

# HYPER PARAMETERS
#INPUT_COLS = ['CHK', 'PWH', 'PDC', 'TWH']
OUTPUT_COLS = ['QTOT']
hidden_layers = 50
#n_epochs = 25
#lr = 0.001
#l2_reg = 0.005
batch_size = 2048
shuffle = True


def print_results(wellnum, net_p, lrate, epochs, layer, regularization, mse_v, mae_v, mape_v, mse_t, mae_t, mape_t):
    print(f'________RESULTS FOR WELL: {wellnum + 1}________')
    print(f'Layers: {layer}')
    print(f'Number epochs: {epochs}')
    print(f'Learning rate: {lrate}')
    print(f'Regularization (l2): {regularization}')
    print(f'Number of model parameters: {sum(p.numel() for p in net_p.parameters())}')

    # print('\nError on validation data')
    # print(f'MSE: {round(mse_v.item(),4)}')
    # print(f'MAE: {round(mae_v.item(),4)}')
    # print(f'MAPE: {round(mape_v.item(),4)} %')

    print('\nError on test data')
    print(f'MSE: {round(mse_t.item(),4)}')
    print(f'MAE: {round(mae_t.item(),4)}')
    print(f'MAPE: {round(mape_t.item(),4)} % \n')


def generate_path(wellnum, features, mape, mae, mse, final):
    mape = str(round(mape, 3))
    mae = str(round(mae, 3))
    mse = str(round(mse, 3))

    features = str(features)[1:-1]
    if not path.exists("results_stuff"):
        mkdir("results_stuff")
    if not path.exists(f'results_stuff/W{wellnum + 1}'):
        mkdir(f'results_stuff/W{wellnum + 1}')

    if final:
        file_path = f'results_stuff/W{wellnum + 1}/W{wellnum + 1} MSE {float(mse):4.4f} MAE {float(mae):.4f} MAPE {float(mape):.4f}' + '.png'
        return file_path

    if not path.exists(f'results_stuff/W{wellnum + 1}/{features}'):
        mkdir(f'results_stuff/W{wellnum + 1}/{features}')

    file_path = f'results_stuff/W{wellnum + 1}/{features}/ W{wellnum + 1} MSE {float(mse):4.4f} MAE {float(mae):.4f} MAPE {float(mape):.4f}' + '.png'
    return file_path


def plot_and_save(wellnum, y, pred, mae, mape, mse, lrate, features, regulatization, epochs, final):
    plt.figure(figsize=(16, 9))
    plt.plot(y.numpy(), label='Missing QTOT')
    plt.plot(pred.detach().numpy(), label='Estimated QTOT')
    plt.title(label=f'Details: well {wellnum + 1} MAE: {mae.item():.4f}, MSE: {mse.item():.4f}, MAPE: {mape.item():.4f} % \n '
               f'Input columns: {features}, \n Learning rate: {lrate}, Epochs: {epochs}, L2-reg: {regulatization}, \n'
               f'Layers: {layers}, Batch size: {batch_size}, Shuffle: {str(shuffle)}, Random seed: {rseed}'
               , fontdict=None, loc='center', pad=None)
    plt.legend()
    plt.xlabel("Samples")
    plt.ylabel("Flow (QTOT)")


    plt.savefig(generate_path(wellnum, features, mape.item(), mae.item(), mse.item(), final), bbox_inches='tight')


    #plt.show()
    plt.clf()


def split_data_by_well(data):
    well1 = data[data.Well == 'W1'].reset_index(drop=True)
    well2 = data[data.Well == 'W2'].reset_index(drop=True)
    well3 = data[data.Well == 'W3'].reset_index(drop=True)
    well4 = data[data.Well == 'W4'].reset_index(drop=True)
    well5 = data[data.Well == 'W5'].reset_index(drop=True)
    return [well1, well2, well3, well4, well5]


def remove_faulty_data(data_init):
    data, datacopy = data_init, data_init
    print("***Filtering***")
    print("Size of data: " + str(len(data)))

    # rule 1
    rule1 = (data['PDC'] < 0)
    print("Samples filtered in rule 1: " + str(rule1.sum()))
    data = data[~rule1]

    # rule 2
    rule2 = (data['PWH'] < 0)
    print("Samples filtered in rule 2: " + str(rule2.sum()))
    data = data[~rule2]

    # rule 3
    rule3 = (data['PWH'] < data['PDC'])
    print("Samples filtered in rule 3: " + str(rule3.sum()))
    data = data[~rule3]

    # rule 4
    rule4 = (data['TWH'] < 0)
    print("Samples filtered in rule 4: " + str(rule4.sum()))
    data = data[~rule4]

    # rule 5
    rule5 = (data['CHK'] < 0.05)
    print("Samples filtered in rule 5: " + str(rule5.sum()))
    data = data[~rule5]

    # rule 6
    rule6 = (data['CHK'] < 0.2) & (data['QTOT'] >= 10)
    print("Samples filtered in rule 6: " + str(rule6.sum()))
    data = data[~rule6]

    # rule 7
    rule7 = (data['CHK'] >= 0.5) & (data['QTOT'] <= 5)
    print("Samples filtered in rule 7: " + str(rule7.sum()))
    data = data[~rule7]

    # skipped samples
    # therules = rule1 | rule2 | rule3 | rule4 | rule5 | rule6 | rule7
    #skipped = datacopy[therules]

    #if len(skipped) > 0:
    #     pd.set_option("display.max_rows", 10000)
    #     print("skipped samples: ")
    #     print(skipped)
    #     pd.set_option("display.max_rows", 7)

    #if len(data_init) - len(skipped) != len(data):
    #    print("Mismatch in data preprocessing")
    print("***Filtering done***\n")
    return data


def shift_data_in_column(data, column, setfillto):
    for index in range(len(data)):
        shifted = data[index][column].shift(periods=1)
        shifted = shifted.fillna(setfillto)
        data[index][column + '_shifted'] = shifted
    return data


def construct_deltap_sqrt(data):
    data['DeltaP2'] = (data['PWH'] - data['PDC']) ** 2
    return data


def remove_low_values(pred, val):
    threshold = 0.5
    # rule_pred = (pred < threshold)
    rule_val = (val < threshold)

    rule = rule_val #| rule_pred
    pred = pred[~rule]
    val = val[~rule]

    return pred, val


def length_of_data(string, data):
    for welln in range(len(data)):
        print("Length of " + string + " of W" + str(welln + 1) + ": " + str(len(data[welln])))


def get_features(index, wellnum):
    feats = []
    if wellnum == 0: # well 1
        feats = [
            [['CHK', 'PWH', 'PDC', 'TWH'],                                                  0.002,  7500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'DeltaP2'],                                       0.002, 10000, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted'],                                  0.003, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted', 'DeltaP2'],                       0.003,  7500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted', 'Z'],                             0.002, 12500, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted', 'Z', 'DeltaP2'],                  0.003, 10000, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted'],                                  0.002, 10000, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'DeltaP2'],                       0.002,  7500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted'],                  0.002, 10000, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted', 'DeltaP2'],       0.003,  7500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted', 'Z'],             0.002,  5000, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted', 'Z', 'DeltaP2'],  0.003, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'Z'],                             0.001,  5000, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'Z', 'DeltaP2'],                  0.003,  5000, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'Z'],                                             0.003, 10000, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'Z', 'DeltaP2'],                                  0.001, 12500, 0.003]]

    elif wellnum == 1: # well 2
        feats=[
            [['CHK', 'PWH', 'PDC', 'TWH'],                                                  0.003, 12500, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'DeltaP2'],                                       0.002,  7500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted'],                                  0.002, 10000, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted', 'DeltaP2'],                       0.001,  7500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted', 'Z'],                             0.003,  7500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted', 'Z', 'DeltaP2'],                  0.001,  7500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted'],                                  0.001,  2500, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'DeltaP2'],                       0.003,  2500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted'],                  0.002,  2500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted', 'DeltaP2'],       0.001,  5000, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted', 'Z'],             0.001,  2500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted', 'Z', 'DeltaP2'],  0.003,  7500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'Z'],                             0.001,  2500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'Z', 'DeltaP2'],                  0.002, 10000, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'Z'],                                             0.003, 10000, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'Z', 'DeltaP2'],                                  0.003,  7500, 0.005]]

    elif wellnum == 2: # well 3
        feats = [
            [['CHK', 'PWH', 'PDC', 'TWH'],                                                  0.002, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'DeltaP2'],                                       0.001, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted'],                                  0.003, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted', 'DeltaP2'],                       0.003, 10000, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted', 'Z'],                             0.001, 10000, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted', 'Z', 'DeltaP2'],                  0.003, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted'],                                  0.002, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'DeltaP2'],                       0.001,  7500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted'],                  0.001, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted', 'DeltaP2'],       0.001, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted', 'Z'],             0.002, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted', 'Z', 'DeltaP2'],  0.002, 10000, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'Z'],                             0.003, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'Z', 'DeltaP2'],                  0.003, 10000, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'Z'],                                             0.003,  7500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'Z', 'DeltaP2'],                                  0.003, 12500, 0.003]]


    elif wellnum == 3: # well 4
        feats = [
            [['CHK', 'PWH', 'PDC', 'TWH'],                                                  0.003, 12500, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'DeltaP2'],                                       0.001, 10000, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted'],                                  0.003, 12500, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted', 'DeltaP2'],                       0.001, 10000, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted', 'Z'],                             0.003, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted', 'Z', 'DeltaP2'],                  0.001, 10000, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted'],                                  0.003, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'DeltaP2'],                       0.002, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted'],                  0.001, 10000, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted', 'DeltaP2'],       0.003, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted', 'Z'],             0.003, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted', 'Z', 'DeltaP2'],  0.003, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'Z'],                             0.003,  7500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'Z', 'DeltaP2'],                  0.003, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'Z'],                                             0.001, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'Z', 'DeltaP2'],                                  0.003, 10000, 0.005]]


    elif wellnum == 4: # well 5
        feats = [
            [['CHK', 'PWH', 'PDC', 'TWH'],                                                  0.002, 12500, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'DeltaP2'],                                       0.003,  7500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted'],                                  0.003, 12500, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted', 'DeltaP2'],                       0.001, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted', 'Z'],                             0.003, 10000, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FGAS_shifted', 'Z', 'DeltaP2'],                  0.003, 12500, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted'],                                  0.002, 12500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'DeltaP2'],                       0.002, 12500, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted'],                  0.003, 10000, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted', 'DeltaP2'],       0.003, 10000, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted', 'Z'],             0.003,  7500, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'FGAS_shifted', 'Z', 'DeltaP2'],  0.003,  5000, 0.003],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'Z'],                             0.002, 10000, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'FOIL_shifted', 'Z', 'DeltaP2'],                  0.002,  5000, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'Z'],                                             0.003, 12500, 0.005],
            [['CHK', 'PWH', 'PDC', 'TWH', 'Z', 'DeltaP2'],                                  0.003, 10000, 0.005]]
            

    else:
        assert well > 4, "No more features :)"
        return None

    return feats[index]


if __name__ == "__main__":
    all_training_data       = pd.read_csv('train.csv', index_col=0)
    all_validation_data     = pd.read_csv('val.csv', index_col=0)
    all_testing_data        = pd.read_csv('test.csv', index_col=0)

    all_training_data       = remove_faulty_data(all_training_data).reset_index(drop=True)
    all_validation_data     = remove_faulty_data(all_validation_data).reset_index(drop=True)
    all_testing_data        = remove_faulty_data(all_testing_data).reset_index(drop=True)
    all_train_plus_val_data = pd.concat([all_training_data, all_validation_data]).reset_index()

    all_training_data       = construct_deltap_sqrt(all_training_data)
    all_validation_data     = construct_deltap_sqrt(all_validation_data)
    all_testing_data        = construct_deltap_sqrt(all_testing_data)
    all_train_plus_val_data = construct_deltap_sqrt(all_train_plus_val_data)

    training_data           = split_data_by_well(all_training_data)
    validation_data         = split_data_by_well(all_validation_data)
    test_data               = split_data_by_well(all_testing_data)
    train_plus_val_data     = split_data_by_well(all_train_plus_val_data)

    length_of_data("Training_data", training_data)
    length_of_data("Validation_data", validation_data)
    length_of_data("Test_data", test_data)
    length_of_data("Training plus validation data", train_plus_val_data)

    training_data = shift_data_in_column(data=training_data, column='FOIL', setfillto=0.333)
    training_data = shift_data_in_column(data=training_data, column='FGAS', setfillto=0.667)
    training_data = shift_data_in_column(data=training_data, column='Z', setfillto=0.8)
    validation_data = shift_data_in_column(data=validation_data, column='FOIL', setfillto=0.333)
    validation_data = shift_data_in_column(data=validation_data, column='FGAS', setfillto=0.667)
    validation_data = shift_data_in_column(data=validation_data, column='Z', setfillto=0.8)
    test_data = shift_data_in_column(data=test_data, column='FOIL', setfillto=0.333)
    test_data = shift_data_in_column(data=test_data, column='FGAS', setfillto=0.667)
    test_data = shift_data_in_column(data=test_data, column='Z', setfillto=0.8)
    train_plus_val_data = shift_data_in_column(data=train_plus_val_data, column='FOIL', setfillto=0.333)
    train_plus_val_data = shift_data_in_column(data=train_plus_val_data, column='FGAS', setfillto=0.667)
    train_plus_val_data = shift_data_in_column(data=train_plus_val_data, column='Z', setfillto=0.8)

    starttot = time.time()
    for well in range(5):
        for i in range(16):  # number of different feature combinations
            features_ = get_features(i, well)
            INPUT_COLS, lr, n_epochs, l2_reg = features_[0], features_[1], int(features_[2]/100), features_[3]
            print("input: ", INPUT_COLS, " lr: ", lr, " epochs: ", n_epochs, " reg: ", l2_reg, "\n")
            start = time.time()
            layers = [len(INPUT_COLS), hidden_layers, hidden_layers, hidden_layers, len(OUTPUT_COLS)]
            print("\n________Starts training of well " + str(well + 1) + "________")
            x_train = torch.from_numpy(train_plus_val_data[well][INPUT_COLS].values).to(torch.float)
            y_train = torch.from_numpy(train_plus_val_data[well][OUTPUT_COLS].values).to(torch.float)

            x_val = torch.from_numpy(validation_data[well][INPUT_COLS].values).float().to(torch.float)
            y_val = torch.from_numpy(validation_data[well][OUTPUT_COLS].values).float().to(torch.float)

            x_test = torch.from_numpy(test_data[well][INPUT_COLS].values).to(torch.float)
            y_test = torch.from_numpy(test_data[well][OUTPUT_COLS].values).to(torch.float)

            train_dataset   = torch.utils.data.TensorDataset(x_train, y_train)
            train_loader    = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
            val_dataset     = torch.utils.data.TensorDataset(x_val, y_val)
            val_loader      = torch.utils.data.DataLoader(val_dataset, batch_size=len(validation_data[well][INPUT_COLS]), shuffle=shuffle)
            test_dataset    = torch.utils.data.TensorDataset(x_val, y_val)
            test_loader     = torch.utils.data.DataLoader(test_dataset, batch_size=len(validation_data[well][INPUT_COLS]), shuffle=shuffle)

            net = NeuralNetwork(layers)
            net = train(net, train_loader, test_loader, n_epochs, lr, l2_reg)

            pred_val  = net(x_val)
            pred_test = net(x_test)

            # remove low values from MAPE such that the MAPE doesn't explode
            pred_val_mape, y_val_mape = remove_low_values(pred_val, y_val)
            pred_test_mape, y_test_mape = remove_low_values(pred_test, y_test)

            mse_val  = torch.mean(torch.pow(pred_val - y_val, 2))
            mae_val  = torch.mean(torch.abs(pred_val - y_val))
            mape_val = torch.mean(torch.abs(torch.div(pred_val_mape - y_val_mape, y_val_mape))) * 100

            mse_test  = torch.mean(torch.pow(pred_test - y_test, 2))
            mae_test  = torch.mean(torch.abs(pred_test - y_test))
            mape_test = torch.mean(torch.abs(torch.div(pred_test_mape - y_test_mape, y_test_mape))) * 100

            print_results(well, net, lr, n_epochs, layers, l2_reg, mse_val, mae_val, mape_val, mse_test, mae_test, mape_test)
            plot_and_save(well, y_test, pred_test, mae_test, mape_test, mse_test, lr, INPUT_COLS, l2_reg, n_epochs, final=True)
            #plot_and_save(well, y_val,  pred_val,  mae_val,  mape_val,  mse_val,  lr, INPUT_COLS, l2_reg, n_epochs, final=False)
            end = time.time()

            print("Time elapsed for well " + str(well + 1) + ": " + str(
                round(end - start, 4)) + " seconds, aka " + str(round((end - start) / 60, 4)) + " minutes.")
            print("____________________________________")

    endtot = time.time()
    print("Total time elapsed: " + str(round((endtot - starttot) / 60, 4)) + " minutes.")
