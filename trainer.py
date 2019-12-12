from scripts.model import ResNext

my_model = ResNext(epochs=50, batchsize=16, learning_rate=5e-4, blocks=3, cardinality=6, depth=48)
my_model.net_init()
my_model.start_training(load_model=False, index=50)
