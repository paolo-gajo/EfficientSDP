import torch
import os
import numpy as np
import sys
import json
from copy import deepcopy
from tqdm.auto import tqdm
from model.utils import build_dataloader, setup_config, build_model, get_args, run_evaluation
from model.utils.sys_utils import save_python_command, save_reproduce_training_cmd
from model.utils.io_utils import save_json
from model.utils.train_utils import get_scheduler, print_params
from model.config import default_cfg, custom_config

def main():

    # get config
    string_args = "" # used for debugging, leave empty for default behavior
    # string_args = "--task_type graph --model_type graph --dataset_name cifar10 --epochs 3 --eval_steps 10000 --eval_samples 0 --batch_size 64 --learning_rate 0.001 --arc_norm 0 --arc_representation_dim 100 --encoder_output_dim 100 --lgi_enc_layers 3 --use_clip_grad_norm 1 --lgi_gat_type base --gat_norm 0"
    args = get_args(string_args=string_args)
    config = setup_config(default_cfg, args=args, custom_config=custom_config)
    print('Config:\n\n', json.dumps(config, indent=4))
    print('Args:\n\n', json.dumps(args, indent=4))
    print(f"Will save to: {config['save_dir']}")
    
    # save config in advance in case training fails
    summary = {'config': config}
    config_path = os.path.join(config['save_dir'], 'config.json')
    save_json(summary, config_path)
    print(f"Config saved to: {config_path}")
    cmd_file = os.path.join(config['save_dir'], 'train_command.txt')
    save_python_command(cmd_file, sys.argv)
    reproduce_training_cmd_file = os.path.join(config['save_dir'], 'full_train_reproduce_cmd.txt')
    save_reproduce_training_cmd(sys.argv[0], config, reproduce_training_cmd_file)

    # data
    dataloader = build_dataloader(config)

    # set `model_start_path` to restart training
    model_start_path = None
    model = build_model(config, model_start_path=model_start_path, verbose=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    scheduler = get_scheduler(optimizer=optimizer,
                            warmup_steps=int(config['train_steps'] * config['warmup_ratio']),
                            train_steps=config['train_steps'],
                            scheduler_type=config['scheduler_type'],
                            use_warmup=config['use_warmup'])

    val_results_list = []
    test_results_list = []
    losses = []
    best_model_state = None
    best_val_metric = -np.inf
    patience_counter = 0

    # we can either train for a number of epochs or steps
    num_epochs = config.get('epochs', 0)
    if num_epochs:
        train_steps = len(dataloader['train'].dataset) * num_epochs // config['batch_size']
    else:
        train_steps = config.get('train_steps', 10000)
    config['eval_steps'] = min(train_steps, config['eval_steps'])

    patience = config.get('patience', 0.3)
    current_step = 0
    unfrozen = False
    train_iter = iter(dataloader['train'])  # create an initial iterator

    with tqdm(total=train_steps, desc="Training Steps") as pbar:
        while current_step < train_steps:
            model.current_step = current_step
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(dataloader['train']) # refill iterator
                batch = next(train_iter)

            model.train()
            model.set_mode('train')
            optimizer.zero_grad()
            loss = model(batch)
            losses.append(loss.item())

            if torch.isnan(loss).item():
                continue

            loss.backward()
            if config['use_clip_grad_norm']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip_norm'])
            
            optimizer.step()
            if config['use_warmup']: scheduler.step()

            current_step += 1
            pbar.update(1)
            pbar.set_description(f"Steps: {current_step}, Loss: {loss.item()}")

            if current_step % config['eval_steps'] == 0:
                print(f'Metrics @ {current_step}:')
                val_results, _ = run_evaluation(model=model,
                                                data_loader=dataloader['val'],
                                                config=config,
                                                label_index_map=config.get('label_index_map', {}),
                                                steps=current_step,)
                # print(f'val F1:\t', json.dumps(val_results, indent=4))
                val_results_list.append(val_results)
                print(f'labeled_val_F1:', [el['parser_labeled_results'].get('F1', None) for el in val_results_list])
                print(f'unlabeled_val_F1:', [el['parser_unlabeled_results'].get('F1', None) for el in val_results_list])
                print(f'val_las:', [el['uas_las_results'].get('las', None) for el in val_results_list])
                save_json(val_results_list, os.path.join(config['save_dir'], "val_results_partial.json"))

                parser_f1 = val_results['parser_unlabeled_results']['F1']

                if config.get('early_stopping', False):
                    if parser_f1 > best_val_metric:
                        print(f'Best model updated at steps {current_step}: ({parser_f1} > {best_val_metric})')
                        best_val_metric = parser_f1
                        best_model_state = deepcopy(model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter * config['eval_steps'] >= train_steps * patience:
                        print("Early stopping triggered.")
                        current_step = train_steps
                        break

                test_results, _ = run_evaluation(model=model,
                                                data_loader=dataloader['test'],
                                                config=config,
                                                label_index_map=config.get('label_index_map', {}),
                                                steps=current_step,)

                test_results_list.append(test_results)
                print(f'labeled_test_F1:', [el['parser_labeled_results'].get('F1', None) for el in test_results_list])
                print(f'unlabeled_test_F1:', [el['parser_unlabeled_results'].get('F1', None) for el in test_results_list])
                print(f'test_las:', [el['uas_las_results'].get('las', None) for el in test_results_list])
                save_json(test_results_list, os.path.join(config['save_dir'], "test_results_partial.json"))
                print('#' * 100)
            
            if current_step >= config['use_gnn_steps'] \
                and not unfrozen \
                and config['parser_type'] in ['gat', 'gat_unbatched'] \
                and config['model_type'] == 'attn':
                unfrozen = model.unfreeze_gnn()
                model.init_gnn_biaffines(optimizer)
                best_model_state = deepcopy(model.state_dict())

    # save best model checkpoint
    if best_model_state is not None and config.get('save_model', False):
        torch.save(best_model_state, config['model_path'])
        print(f"Model saved at: {config['model_path']}")

    # save validation results and configuration details
    os.remove(os.path.join(config['save_dir'], "val_results_partial.json"))
    os.remove(os.path.join(config['save_dir'], "test_results_partial.json"))
    save_json(val_results_list, os.path.join(config['save_dir'], "val_results_series.json"))
    save_json(test_results_list, os.path.join(config['save_dir'], "test_results_series.json"))
    save_json(losses, os.path.join(config['save_dir'], "losses.json"))
    if hasattr(dataloader['train'].dataset, 'label_index_map'):
        save_json(dataloader['train'].dataset.label_index_map, os.path.join(config['save_dir'], 'labels.json'))

    # final evaluation on validation and test sets
    if config.get('save_model', False):
        model.load_state_dict(best_model_state)
        val_results, benchmark_metrics = run_evaluation(
            model, dataloader['val'], config, config.get('label_index_map', {}),
        )
        save_json(val_results, os.path.join(config['save_dir'], f"val_results.json"))
        save_json(benchmark_metrics, os.path.join(config['save_dir'], 'val_results_benchmark.json'))
        print('Validation results:', val_results)

        test_results, benchmark_metrics = run_evaluation(
            model, dataloader['test'], config, config.get('label_index_map', {}),
        )
        save_json(test_results, os.path.join(config['save_dir'], f"test_results.json"))
        save_json(benchmark_metrics, os.path.join(config['save_dir'], 'test_results_benchmark.json'))
        print('Test results:', test_results)

    summary = {'config': config, 'val_results': val_results, 'test_results': test_results}
    save_json(summary, os.path.join(config['save_dir'], 'config.json'))

if __name__ == "__main__":
    main()