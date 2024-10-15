from Detectify import Train

train = Train()

train.start(dataset="./dataset/data.yaml",
            epochs=10,
            batch_size=64,
            save_per_epochs=5,
            cache=True)