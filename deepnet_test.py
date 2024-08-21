from deep_learner import *


def main():
    model_path = 'results/deepnet/best_model_small.torch'
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
        shuffle=True,
        pin_memory=True,
    )
    datasamp = testset[0][0]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_info = torch.load(model_path)

    model = FCNet(len(datasamp), 1, layers=[20, 20, 20])
    # print(model)
    # print([model_info['model'][m].shape for m in model_info['model']])
    model.load_state_dict(model_info['model'])
    model.to(device)
    model.eval()
    tot_preds = []
    pbar = tqdm(total=len(testload), file=sys.stdout)
    desc = f'Test Data'
    pbar.set_description(desc)
    with torch.no_grad():
        for iter, (batch,) in enumerate(testload):
            batch = batch.to(device)
            pred = model(batch)
            tot_preds.append(F.sigmoid(pred) > 0.5)
            pbar.update(1)
    pbar.close()

    tot_preds = torch.cat(tot_preds, 0).cpu().numpy()
    data['Transported'] = tot_preds.astype(bool)
    savedata = data[['PassengerId', 'Transported']]
    savedata.to_csv('results/deepnet_submission.csv', index=False)


if __name__ == '__main__':
    main()