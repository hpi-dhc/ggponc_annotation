from datasets import load_dataset, load_metric, Sequence, ClassLabel
import transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments, pipeline, DataCollatorForTokenClassification, trainer_utils
import numpy as np
import pandas as pd
from pathlib import Path
import os, sys
import json
import huggingface_utils
import error_analysis
import argparse
from torch import nn
import logging
from rich.console import Console
import wandb
from tqdm import tqdm

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, ListConfig

log = logging.getLogger(__name__)

cp_path = Path('checkpoints')
        
class LogHandler(transformers.TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            log.info(logs)

def get_path(path):
    return Path(hydra.utils.to_absolute_path(path))

def get_train_args(
    cp_path, run_name, report_to, batch_size, learning_rate, num_train_epochs, weight_decay, save_steps, lr_scheduler_type, fp16, warmup_ratio, keep_all_checkpoints, label_smoothing_factor, resume_from_checkpoint, gradient_checkpointing, **kwargs):
    log.info(run_name)
    args = TrainingArguments(
        str(cp_path),
        evaluation_strategy = "epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        save_strategy="epoch",
        fp16=fp16,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        save_steps=save_steps,
        save_total_limit=None if keep_all_checkpoints else 2,
        overwrite_output_dir=True,
        report_to=report_to,
        load_best_model_at_end=True,
        metric_for_best_model="eval_overall_f1",
        gradient_checkpointing=gradient_checkpointing,
        label_smoothing_factor=label_smoothing_factor,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    return args

def run(config, run_name, sweep_name):
    log.info(f'Fixing random seed {config.random_seed}')
    trainer_utils.set_seed(config.random_seed)
    
    transformers.logging.disable_default_handler()
    
    log.info(OmegaConf.to_yaml(config))
    log.info('Running in: ' + os.getcwd())
    
    cp_path = Path(config.checkpoint_path)
    base_model_checkpoint = config["base_model_checkpoint"]
    
    args = get_train_args(cp_path, run_name, 'wandb', **config, resume_from_checkpoint=None)

    tokenizer = AutoTokenizer.from_pretrained(base_model_checkpoint)#, add_prefix_space=True)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    
    task = config.get('task', None)
        
    dataset, tags = huggingface_utils.load_custom_dataset(
        train=get_path(config.train_dataset), 
        dev=get_path(config.dev_dataset), 
        test=get_path(config.test_dataset),
        tag_strings=task)
   
    label_aligner = huggingface_utils.LabelAligner(tokenizer)        
    dataset = dataset.map(lambda e: label_aligner.tokenize_and_align_labels(e, config.label_all_tokens), batched=True)
        
    id2label = dict(enumerate(tags))
    
    log.info(id2label)
    log.info(dataset)
            
    if config.checkpoint:
        log.info(f"Starting from checkpoint: {config.checkpoint}")
        model = AutoModelForTokenClassification.from_pretrained(checkpoint)
    elif 'pre_trained_model_file' in config and config.pre_trained_model_file:
        pre_trained_model_file = get_path(config['pre_trained_model_file'])
        log.info(f"Loading model file: {pre_trained_model_file}")
        model = AutoModelForTokenClassification.from_pretrained(pre_trained_model_file)
    else:
        log.info(f"Starting from base model: {base_model_checkpoint}")
        model = AutoModelForTokenClassification.from_pretrained(
            base_model_checkpoint,
            num_labels=len(tags), 
            id2label=id2label
        )
    
    if config.link:
        link_name = "latest" if type(config.link) is bool else config.link
        link_base = get_path('..') / 'models' / config.name
        link_base.mkdir(exist_ok=True)
        link_src = link_base / link_name
        log.info(f'Linking {link_src} to this run')
        link_src.unlink(missing_ok=True)
        os.symlink(os.getcwd(), link_src, target_is_directory=True)
        
    with wandb.init(project=config.wand_db_project, 
           name=run_name,
           tags=[Path(config.train_dataset).name],
           group=config.name,
           reinit=True) as run:
    
        wandb.log(OmegaConf.to_container(config))
        wandb.log({'hydra_sweep' : sweep_name})
 
        data_collator = DataCollatorForTokenClassification(tokenizer)
        tr = Trainer(
            model,
            args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["dev"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=huggingface_utils.compute_metrics(tags, True),
        )

        tr.add_callback(LogHandler)

        train_result = tr.train()

        log.info('**** Evaluate ****')
        
        eval_metrics = tr.evaluate()
        
        log.info(eval_metrics)
        wandb.log(eval_metrics)
        
        token_test_ds = dataset["test"].map(lambda e: e)
        
        test_metrics = huggingface_utils.eval_on_test_set(dataset["test"], tr, tokenizer, "test")
        
        log.info(test_metrics)
        wandb.log(test_metrics)

        os.symlink(tr.state.best_model_checkpoint, 'best_cp', target_is_directory=True)
        
        if "output_file" in config:
            log.info(f"Saving model to: {config['output_file']}")
            tr.model.save_pretrained(config["output_file"])
        
@hydra.main(config_path='.')
def main(config: DictConfig) -> None:    
    hydra_conf = HydraConfig.get()
        
    run_name = f"{config.name}_{hydra_conf.job.override_dirname}"
    
    if 'id' in hydra_conf.job:
        # In a sweep
        sweep_name = hydra_conf.sweep.dir
        log.info("Running in a parameter sweep")
    else:
        sweep_name = config.date_run
        
    log.info(f"Grouping by {sweep_name}")
        
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda)
    
    os.environ["WANDB_PROJECT"] = config.wand_db_project
    
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    log.info(f'Running on CUDA device: {config.cuda}')    

    run(config, run_name, sweep_name)

if __name__ == "__main__":
    main()
