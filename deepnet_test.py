from deep_learner import *


def main():
    model_name = '100bnlndrmishsch'
    # model_path = 'results/deepnet/deepnet_{}_e375.torch'.format(model_name)
    model_path = 'results/deepnet/best_model_{}.torch'.format(model_name)
    # model_path = 'results/deepnet/deepnet_e2000_100bnlndrmishsch.torch'
    batch_size = 100
    use_luxery = True
    # ['PassengerId', 'HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Name', 'Transported']
    data = pd.read_csv("data/test.csv") 
    tdata = process_data(data, use_luxery=use_luxery)

    tdata = tensor_process(tdata)
    testset = torch.utils.data.TensorDataset(tdata)
    testload = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    datasamp = testset[0][0]

    validation_data = pd.read_csv("data/valid.csv")
    vdata = process_data(validation_data, use_luxery=use_luxery)
    vdata = tensor_process(vdata)
    vtargets = torch.tensor(validation_data['Transported'], dtype=torch.float32).reshape(-1, 1)
    validset = torch.utils.data.TensorDataset(vdata, vtargets)
    validload = DataLoader(
        validset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )
    validation_data = pd.read_csv("data/valid.csv")
    vdata = process_data(validation_data, use_luxery=use_luxery)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_info = torch.load(model_path)

    model = FCNet(len(datasamp), 1, layers=[100, 100, 100, 100, 100], activation='mish', batchnorm=True, layernorm=True, dropout=0.5)
    # model = FCNet(len(datasamp), 1, layers=[50, 50, 50, 50, 50], activation='elu', batchnorm=False, layernorm=True, dropout=0.5)
    # model = FCNet(len(datasamp), 1, layers=[200, 200, 200, 200, 200], activation='mish', batchnorm=True, layernorm=True, dropout=0.5)
    # print(model)
    # print([model_info['model'][m].shape for m in model_info['model']])
    model.load_state_dict(model_info['model'])
    model.to(device)
    print(model)

    tot_preds = []
    with torch.no_grad():
        model.eval()
        vloss, vacc = model.evaluate(validload)
        print('vloss:', vloss)
        print('vacc:', vacc)
        pbar = tqdm(total=len(testload), file=sys.stdout)
        desc = f'Test Data'
        pbar.set_description(desc)
        for iter, (batch,) in enumerate(testload):
            batch = batch.to(device)
            pred = model(batch)
            # import pdb; pdb.set_trace()
            tot_preds.append(F.sigmoid(pred) > 0.5)
            pbar.update(1)
        pbar.close()

    tot_preds = torch.cat(tot_preds, 0).cpu().numpy()
    data['Transported'] = tot_preds.astype(bool)
    savedata = data[['PassengerId', 'Transported']]
    savedata.to_csv('results/deepnet_submission.csv', index=False)


if __name__ == '__main__':
    main()