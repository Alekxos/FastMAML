import torch
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from torch.optim import Adam
import util
from pathlib import Path
from helper import *

def create_output_directory(output_dir, custom_savename, meta_train=False, meta_val=False):
    if not path.exists(output_dir):
        os.mkdir(output_dir)
    current_run_savedir = str(Path(output_dir) / custom_savename)
    create_directory(current_run_savedir)
    # Create folder for saved models
    model_path = str(Path(current_run_savedir) / "models")
    create_directory(model_path)
    if meta_train:
        meta_train_dir = str(Path(current_run_savedir) / "meta_train")
        create_directory(meta_train_dir)
    if meta_val:
        meta_val_dir = str(Path(current_run_savedir) / "meta_validation")
        create_directory(meta_val_dir)
    return current_run_savedir

def save_metric_results(current_run_savedir, metrics, meta_train):
    tracked_support_losses, tracked_support_accuracies, tracked_query_losses, tracked_query_accuracies = metrics
    # Save metrics to file
    subfolder = "meta_train" if meta_train else "meta_validation"
    paths = [str(Path(current_run_savedir) / subfolder / save_suffix) for save_suffix in ['support_loss.npy',
                                                                                          'support_accuracy.npy',
                                                                                          'query_loss.npy',
                                                                                          'query_accuracy.npy']]
    np.save(f'{paths[0]}', np.array(tracked_support_losses))
    np.save(f'{paths[1]}', np.array(tracked_support_accuracies))
    np.save(f'{paths[2]}', np.array(tracked_query_losses))
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

def meta_iteration(model, config, meta_image_batch, meta_label_batch):
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
                                           cuda=config.cuda) if inner_loop_num != 0 else model.forward(support_input,
                                                                                                cuda=config.cuda)
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
            # TODO: Replace standard SGD inner loop update with more powerful optimizer ex. Adam
            adapted_weights = OrderedDict((name, param - config.inner_lr * grad) for ((name, param), grad) in
                                          zip(adapted_weights.items(), inner_grads))

            # Track initial performance on support set
            if inner_loop_num == 0:
                support_losses.append(support_loss)
                support_accuracies.append(support_accuracy.item())
            if inner_loop_num == config.num_inner_updates - 1:
                # Evaluate output and loss on query (inner test) dataset
                query_output = model.forward(query_input, adapted_weights, cuda=config.cuda)
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
    return (support_losses, support_accuracies, query_losses, query_accuracies, meta_gradients)

def save_model(model, config, meta_step):
    save_path = Path(config.output_dir) / config.custom_savename / "models" / f"model_epoch_{meta_step}"
    print(f"Saving model at path {save_path}")
    torch.save(model.state_dict(), str(save_path))

def meta_train(model, data_generator, meta_train_iterations, config):
    # Assign internal constants
    print_interval = 10
    save_interval = 10
    validate_interval = 1

    # Instantiate metric data structures for tracking
    tracked_support_losses, tracked_support_accuracies, tracked_query_losses, tracked_query_accuracies = [], [], [], []
    if config.meta_validate:
        tracked_meta_val_losses, tracked_meta_val_accuracies = [], []

    meta_optimizer = Adam(model.parameters(), lr=config.meta_lr)
    for meta_step in range(meta_train_iterations):
        print(f"Meta-Step {meta_step}")
        # Sample meta batch of data and samples for model initialization
        (meta_image_batch, meta_label_batch) = data_generator.sample_batch("meta_train", config.meta_batch_size,
                                                                           shuffle=True)
        sample_input, sample_label = meta_image_batch[0, 0, 0, :], meta_label_batch[0, 0, 0]
        sample_input = torch.tensor(sample_input).view(-1, 1, config.img_size, config.img_size)

        support_losses, support_accuracies, query_losses, query_accuracies, meta_gradients = meta_iteration(model, config, meta_image_batch, meta_label_batch)

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
            save_metric_results(config.current_run_savedir, metrics_snapshot, meta_train=True)
            # Save model
            save_model(model, config, meta_step)

            # Remove outdated model saves
            old_model_path = Path(config.output_dir) / config.custom_savename / "models" / f"model_epoch_{meta_step - 3 * save_interval}"
            if path.exists(old_model_path):
                os.remove(str(old_model_path))
        if meta_step % validate_interval == 0 and config.meta_validate:
            _, _, validation_query_loss, validation_query_accuracy = meta_validate(model, data_generator, config, cuda=config.cuda)
            tracked_meta_val_losses.append(validation_query_loss)
            tracked_meta_val_accuracies.append(validation_query_accuracy)
            # Save metrics for this meta-epoch
            metrics_snapshot = ([], [], tracked_meta_val_losses, tracked_meta_val_accuracies)
            save_metric_results(config.current_run_savedir, metrics_snapshot, meta_train=False)
            print(f"\n\nMeta-Validation Loss: {validation_query_loss}, Accuracy: {validation_query_accuracy}")
        else:
            tracked_meta_val_losses.append(0)
            tracked_meta_val_accuracies.append(0)

    return model

def meta_validate(model, data_generator, config, cuda=False):
    # Sample meta-batch of data and samples for model initialization
    (meta_image_batch, meta_label_batch) = data_generator.sample_batch("meta_val", config.meta_batch_size,
                                                                       shuffle=True)
    _, sample_label = meta_image_batch[0, 0, 0, :], meta_label_batch[0, 0, 0]

    support_losses, support_accuracies, query_losses, query_accuracies, _ = meta_iteration(model, config,
                                                                                                        meta_image_batch,
                                                                                                        meta_label_batch)

    # Update metrics list
    validation_support_loss = np.mean([support_loss.item() for support_loss in support_losses])
    validation_support_accuracy = np.mean(support_accuracies)
    validation_query_loss = np.mean([query_loss.item() for query_loss in query_losses])
    validation_query_accuracy = np.mean(query_accuracies)

    return (validation_support_loss, validation_support_accuracy, validation_query_loss, validation_query_accuracy)
