from load_data import DataGenerator
import click
from models.MatchingNetworkCNN import MatchingNetworkCNN
import time
from maml import *


class Config:
    def __init__(self, dataset, meta_train_iterations, meta_lr, inner_lr,
                 meta_batch_size, img_size, num_channels, meta_train,
                 meta_validate, k_shot, n_way, data_path, num_inner_updates,
                 output_dir, load_model_path, cuda):
        self.dataset = dataset
        self.meta_train_iterations = meta_train_iterations
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.meta_batch_size = meta_batch_size
        self.img_size = img_size
        self.num_channels = num_channels
        self.meta_train = meta_train
        self.meta_validate = meta_validate
        self.k_shot = k_shot
        self.n_way = n_way
        self.data_path = data_path
        self.num_inner_updates = num_inner_updates
        self.output_dir = output_dir
        self.load_model_path = load_model_path
        self.cuda = cuda

@click.command()
@click.argument('dataset', type=str, default='omniglot')
@click.option('--meta_train_iterations', type=int, default=15000, help='number of meta-training iterations')
@click.option('--meta_lr', type=float, default=0.001, help='outer loop learning rate')
@click.option('--inner_lr', type=float, default=0.04, help='inner loop learning rate')
@click.option('--meta_batch_size', type=int, default=32, help='number of tasks sampled for a single meta-step')
@click.option('--img_size', type=int, default=28, help='height=width of input images')
@click.option('--num_channels', type=int, default=1, help='number of channels in input images')
@click.option('--m_train', type=bool, default=True, help='perform meta training')
@click.option('--m_validate', type=bool, default=True, help='perform meta validation')
@click.option('--k_shot', type=int, default=1, help='number of examples used for inner gradient update training (K '
                                                    'for K-shot learning)')
@click.option('--n_way', type=int, default=5,
              help='number of classes used in classification (e.g. 5-way classification)')
@click.option('--data_path', type=str, default='./datasets/omniglot_resized', help='path to the dataset')
@click.option('--num_inner_updates', type=int, default=1, help='number of inner gradient updates during inner loop training')
@click.option('--output_dir', type=str, default='./output', help='directory to store output, including metrics and model')
@click.option('--load_model_path', type=str, default='.', help='path to periodically save models')
@click.option('--cuda', type=bool, default=False, help='use CUDA while meta training (or not)')

def main(dataset, meta_train_iterations, meta_lr, inner_lr, meta_batch_size, img_size, num_channels,
         m_train, m_validate, k_shot, n_way, data_path, num_inner_updates, output_dir, load_model_path, cuda):
    # Pack configurable parameters into single object
    config = Config(dataset, meta_train_iterations, meta_lr, inner_lr, meta_batch_size, img_size, num_channels, m_train,
                    m_validate, k_shot, n_way, data_path, num_inner_updates, output_dir, load_model_path, cuda)
    # Create engine to sample data from
    data_generator = DataGenerator(n_way, k_shot * 2, n_way, k_shot * 2, config={'data_folder': data_path})
    # Instantiate base model (meta parameters) and optimizer
    model = MatchingNetworkCNN(n_way).double()
    if config.load_model_path != ".":
        if path.exists(load_model_path):
            model.load_state_dict(torch.load(config.load_model_path))
            model.eval()
            print(f"Saved model found at path {config.load_model_path}.\nLoading model...")
        else:
            print(f"Saved model not found at path {config.load_model_path}.\nCreating model from scratch...")
    # Create output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    config.custom_savename = f"{config.dataset}_K_{config.k_shot}_N_{config.n_way}_mlr_{config.meta_lr}_ilr_{config.inner_lr}_{timestamp}"
    config.current_run_savedir = create_output_directory(config.output_dir, config.custom_savename, config.meta_train, config.meta_validate)

    if m_train:
        model = meta_train(model, data_generator, meta_train_iterations, config)

    if m_validate:
        _, _, validation_query_loss, validation_query_accuracy = meta_validate(model, data_generator, config)
        print(f"\n\nFinal Meta-Validation Loss: {validation_query_loss}, Accuracy: {validation_query_accuracy}")

if __name__ == '__main__':
    main()

