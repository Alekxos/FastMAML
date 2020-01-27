import torch
from load_data import DataGenerator
import click
from models.MatchingNetworkCNN import MatchingNetworkCNN
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch.optim import SGD, Adam
import util
from os import path
import os
import time
from pathlib import Path
from pydantic import BaseModel

class Config:
    def __init__(self, dataset, meta_train_iterations, meta_lr, inner_lr,
                 meta_batch_size, img_size, num_channels, meta_train,
                 meta_validate, k_shot, n_way, data_path, num_inner_updates,
                 output_dir, load_model_path):
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

def create_directory(dir_path):
    if not path.exists(dir_path):
        os.mkdir(dir_path)

def create_output_directory(output_dir, custom_savename):
    if not path.exists(output_dir):
        os.mkdir(output_dir)
    current_run_savedir = str(Path(output_dir) / custom_savename)
    create_directory(current_run_savedir)
    # Create folder for saved models
    model_path = str(Path(current_run_savedir) / "models")
    create_directory(model_path)
    return current_run_savedir

def save_metric_results(current_run_savedir, metrics, meta_train):
    tracked_support_losses, tracked_support_accuracies, tracked_query_losses, tracked_query_accuracies = metrics
    # Save metrics to file
    subfolder = "meta_train" if meta_train else "meta_validation"
    paths = [str(Path(current_run_savedir) / subfolder / save_suffix) for save_suffix in ['support_loss.npy',
                                                                                          'support_accuracy.npy',
                                                                                          'query_loss.npy',
                                                                                          'query_accuracy.npy']]
    if tracked_support_losses != []:
        np.save(f'{paths[0]}', np.array(tracked_support_losses))
    if tracked_support_accuracies != []:
        np.save(f'{paths[1]}', np.array(tracked_support_accuracies))
    if tracked_query_losses != []:
        np.save(f'{paths[2]}', np.array(tracked_query_losses))
    if tracked_query_accuracies != []:
        np.save(f'{paths[3]}', np.array(tracked_query_accuracies))

def parse_images_labels(image_batch, label_batch, k_shot, n_way, img_size):
    # Split into support/query
    support_input, query_input = image_batch[:, :k_shot, :], image_batch[:, k_shot:, :]
    support_labels, query_labels = label_batch[:, :k_shot, :], label_batch[:, k_shot:, :]
    # Reshape support/query and convert to Tensor
    support_labels = torch.argmax(torch.tensor(support_labels), dim=-1).view(n_way * k_shot)
    query_labels = torch.argmax(torch.tensor(query_labels), dim=-1).view(n_way * k_shot)
    support_input = torch.tensor(support_input).view(-1, 1, img_size, img_size)
    query_input = torch.tensor(query_input).view(-1, 1, img_size, img_size)
    return (support_input, support_labels, query_input, query_labels)

def update_meta_parameters(model, meta_gradients, meta_optimizer, sample_input, sample_label, n_way):
    sample_output = model.forward(sample_input).view(1, n_way)
    sample_label = torch.argmax(torch.tensor(sample_label), dim=-1).view(1)
    # print(f"sample output: {sample_output.shape}, sample label: {sample_label.shape}")
    loss = F.cross_entropy(sample_output, sample_label)
    meta_gradient_dict = {k: sum(d[k] for d in meta_gradients) for k in meta_gradients[0].keys()}
    hooks = []
    for (k, v) in model.named_parameters():
        def get_closure():
            key = k

            def replace_grad(_):
                return meta_gradient_dict[key]

            return replace_grad

        hooks.append(v.register_hook(get_closure()))
    # Compute grads for current step, replace with summed gradients as defined by hook
    meta_optimizer.zero_grad()
    loss.backward()
    # Update the net parameters with the accumulated gradient according to optimizer
    meta_optimizer.step()
    # Remove the hooks before next training phase
    for h in hooks:
        h.remove()

def meta_train(model, data_generator, meta_train_iterations, config, cuda=False):
    # Assign internal constants
    print_interval = 10
    save_interval = 10
    validate_interval = 50

    # Instantiate metric data structures for tracking
    tracked_support_losses, tracked_support_accuracies, tracked_query_losses, tracked_query_accuracies = [], [], [], []
    if config.meta_validate:
        tracked_meta_val_losses, tracked_meta_val_accuracies = [], []

    meta_optimizer = Adam(model.parameters(), lr=config.meta_lr)
    for meta_step in range(meta_train_iterations):
        print(f"Meta-Step {meta_step}")
        # Sample meta-batch of data and samples for model initialization
        (meta_image_batch, meta_label_batch) = data_generator.sample_batch("meta_train", config.meta_batch_size, shuffle=True)
        sample_input, sample_label = meta_image_batch[0, 0, 0, :], meta_label_batch[0, 0, 0]
        sample_input = torch.tensor(sample_input).view(-1, 1, config.img_size, config.img_size)
        # Record results of inner loop training
        support_losses, support_accuracies, query_losses, query_accuracies, meta_gradients = [], [], [], [], []
        # TODO: Is following for loop parallelizable (akin to tf.map_fn)?
        for task_index in range(config.meta_batch_size):
            image_batch, label_batch = meta_image_batch[task_index], meta_label_batch[task_index]

            # Split input image and label batches into support/query batches
            support_input, support_labels, query_input, query_labels = \
                parse_images_labels(image_batch, label_batch, config.k_shot, config.n_way, config.img_size)

            # Instantiate adapted model parameters and optimizer
            adapted_weights = OrderedDict((name, param) for (name, param) in model.named_parameters())

            # Update adapted model based on gradients from support dataset
            for inner_loop_num in range(config.num_inner_updates):
                # Forward input batch through adapted models
                support_output = model.forward(support_input, adapted_weights,
                                               cuda=cuda) if inner_loop_num != 0 else model.forward(support_input,
                                                                                                     cuda=cuda)
                support_output = support_output.view(config.n_way * config.k_shot, config.n_way)

                # Evaluate support (inner train) loss and accuracy
                support_loss = F.cross_entropy(support_output, support_labels)
                support_predictions = torch.argmax(support_output, dim=-1)
                support_accuracy = util.accuracy(support_predictions, support_labels)

                # Update adapted model
                if inner_loop_num == 0:
                    inner_grads = torch.autograd.grad(support_loss, model.parameters(), create_graph=True)
                else:
                    inner_grads = torch.autograd.grad(support_loss, adapted_weights.values(), create_graph=True)
                adapted_weights = OrderedDict((name, param - config.inner_lr * grad) for ((name, param), grad) in
                                              zip(adapted_weights.items(), inner_grads))

                # Track initial performance on support set
                if inner_loop_num == 0:
                    support_losses.append(support_loss)
                    support_accuracies.append(support_accuracy.item())
                if inner_loop_num == config.num_inner_updates - 1:
                    # Evaluate output and loss on query (inner test) dataset
                    query_output = model.forward(query_input, adapted_weights, cuda=cuda)
                    query_output = query_output.view(config.n_way * config.k_shot, config.n_way)
                    query_loss = F.cross_entropy(query_output, query_labels)
                    query_losses.append(query_loss)
                    # Keeping adapted model fixed, evaluate meta-gradients using query dataset
                    raw_task_meta_gradients = torch.autograd.grad(query_loss, model.parameters())
                    task_meta_gradients = {name: g for ((name, _), g) in
                                           zip(model.named_parameters(), raw_task_meta_gradients)}
                    meta_gradients.append(task_meta_gradients)
            # Also evaluate query accuracy to average over tasks and store later
            query_predictions = torch.argmax(query_output, dim=-1)
            query_accuracy = util.accuracy(query_predictions, query_labels)
            query_accuracies.append(query_accuracy)

        # Update model meta-parameters
        update_meta_parameters(model, meta_gradients, meta_optimizer, sample_input, sample_label, config.n_way)

        # Update metrics list
        tracked_support_losses.append(np.mean([support_loss.item() for support_loss in support_losses]))
        tracked_support_accuracies.append(np.mean(support_accuracies))
        tracked_query_losses.append(np.mean([query_loss.item() for query_loss in query_losses]))
        tracked_query_accuracies.append(np.mean(query_accuracies))
        if meta_step % print_interval == 0:
            print(
                f"Iteration {meta_step}: pre-inner-loop meta-train accuracy: {tracked_support_accuracies[-1]}, post-inner-loop meta-train accuracy: {tracked_query_accuracies[-1]}")
            print(
                f"Iteration {meta_step}: pre-inner-loop meta-train loss: {tracked_support_losses[-1]}, post-inner-loop meta-train loss: {tracked_query_losses[-1]}")
        if meta_step % save_interval == 0:
            # Save metrics for this meta-epoch
            metrics_snapshot = (tracked_support_losses, tracked_support_accuracies, tracked_query_losses, tracked_query_accuracies)
            save_metric_results(config.current_run_savedir, metrics_snapshot)
            # Save model
            save_path = Path(config.output_dir) / config.custom_savename / "models" / f"model_epoch_{meta_step}"
            print(f"Saving model at path {save_path}")
            torch.save(model.state_dict(), str(save_path))
            # Remove outdated model saves
            old_model_path = Path(config.output_dir) / config.custom_savename / "models" / f"model_epoch_{meta_step - 3 * save_interval}"
            if path.exists(old_model_path):
                os.remove(str(old_model_path))
        if meta_step % validate_interval == 0:
            _, _, validation_query_loss, validation_query_accuracy = meta_validate(model, data_generator, config, cuda=cuda)
            tracked_meta_val_losses.append(validation_query_loss)
            tracked_meta_val_accuracies.append(validation_query_accuracy)
            # Save metrics for this meta-epoch
            metrics_snapshot = ([], [], tracked_meta_val_losses, tracked_meta_val_accuracies)
            save_metric_results(config.current_run_savedir, metrics_snapshot)
            print(f"\n\nMeta-Validation Loss: {validation_query_loss}, Accuracy: {validation_query_accuracy}")
        else:
            tracked_meta_val_losses.append(0)
            tracked_meta_val_accuracies.append(0)

    return model

def meta_validate(model, data_generator, config, cuda=False):
    # Sample meta-batch of data and samples for model initialization
    (meta_image_batch, meta_label_batch) = data_generator.sample_batch("meta_val", config.meta_batch_size,
                                                                       shuffle=True)
    sample_input, sample_label = meta_image_batch[0, 0, 0, :], meta_label_batch[0, 0, 0]
    sample_input = torch.tensor(sample_input).view(-1, 1, config.img_size, config.img_size)

    # Record results of inner-loop training
    support_losses, support_accuracies, query_losses, query_accuracies = [], [], [], []

    for task_index in range(config.meta_batch_size):
        image_batch, label_batch = meta_image_batch[task_index], meta_label_batch[task_index]

        # Split input image and label batches into support/query batches
        support_input, support_labels, query_input, query_labels = \
            parse_images_labels(image_batch, label_batch, config.k_shot, config.n_way, config.img_size)

        # Instantiate adapted model parameters and optimizer
        adapted_weights = OrderedDict((name, param) for (name, param) in model.named_parameters())

        # Update adapted model based on gradients from support dataset
        for inner_loop_num in range(config.num_inner_updates):
            # Forward input batch through adapted models
            support_output = model.forward(support_input, adapted_weights,
                                           cuda=cuda) if inner_loop_num != 0 else model.forward(support_input,
                                                                                                cuda=cuda)
            support_output = support_output.view(config.n_way * config.k_shot, config.n_way)

            # Evaluate support (inner train) loss and accuracy
            support_loss = F.cross_entropy(support_output, support_labels)
            support_predictions = torch.argmax(support_output, dim=-1)
            support_accuracy = util.accuracy(support_predictions, support_labels)

            # Update adapted model
            if inner_loop_num == 0:
                inner_grads = torch.autograd.grad(support_loss, model.parameters(), create_graph=True)
            else:
                inner_grads = torch.autograd.grad(support_loss, adapted_weights.values(), create_graph=True)
            adapted_weights = OrderedDict((name, param - config.inner_lr * grad) for ((name, param), grad) in
                                          zip(adapted_weights.items(), inner_grads))

            # Track initial performance on support set
            if inner_loop_num == 0:
                support_losses.append(support_loss)
                support_accuracies.append(support_accuracy.item())
            if inner_loop_num == config.num_inner_updates - 1:
                # Evaluate output and loss on query (inner test) dataset
                query_output = model.forward(query_input, adapted_weights, cuda=cuda)
                query_output = query_output.view(config.n_way * config.k_shot, config.n_way)
                query_loss = F.cross_entropy(query_output, query_labels)
                query_losses.append(query_loss)

        # Also evaluate query accuracy to average over tasks and store later
        query_predictions = torch.argmax(query_output, dim=-1)
        query_accuracy = util.accuracy(query_predictions, query_labels)
        query_accuracies.append(query_accuracy)

    # Update metrics list
    validation_support_loss = np.mean([support_loss.item() for support_loss in support_losses])
    validation_support_accuracy = np.mean(support_accuracies)
    validation_query_loss = np.mean([query_loss.item() for query_loss in query_losses])
    validation_query_accuracy = np.mean(query_accuracies)

    return (validation_support_loss, validation_support_accuracy, validation_query_loss, validation_query_accuracy)


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
@click.option('--load_model_path', type=str, default='./saved_models', help='path to periodically save models')
@click.option('--cuda', type=bool, default=False, help='use CUDA while meta training (or not)')

def main(dataset, meta_train_iterations, meta_lr, inner_lr, meta_batch_size, img_size, num_channels,
         m_train, m_validate, k_shot, n_way, data_path, num_inner_updates, output_dir, load_model_path, cuda):
    # Pack configurable parameters into single object
    config = Config(dataset, meta_train_iterations, meta_lr, inner_lr, meta_batch_size, img_size, num_channels, m_train,
                    m_validate, k_shot, n_way, data_path, num_inner_updates, output_dir, load_model_path)
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
    config.current_run_savedir = create_output_directory(config.output_dir, config.custom_savename)

    if m_train:
        model = meta_train(model, data_generator, meta_train_iterations, config)

    if m_validate:
        _, _, validation_query_loss, validation_query_accuracy = meta_validate(model, data_generator, config,
                                                                               cuda=cuda)
        print(f"\n\nFinal Meta-Validation Loss: {validation_query_loss}, Accuracy: {validation_query_accuracy}")

if __name__ == '__main__':
    main()

