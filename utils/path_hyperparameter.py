class Path_Hyperparameter:
    load_epoch=0
    mytime='time1'
    evaluate_epoch: int = 0
    load: str =False
    arf=2
    beta=0.2

    save_interval: int = 20
    epochs: int = 100
    dataset_name = 'whu'

    random_seed = 42

    warm_up_step = 500


    save_dir=rf'.\run\{dataset_name}\{mytime}/'
    logFile=r'trainValLog.txt'
    batch_size: int = 1
    inference_ratio = 2
    learning_rate: float = 2e-4
    factor = 0.1
    patience = 12

    weight_decay: float = 1e-3
    amp: bool = True
    max_norm: float = 20
    stage_epoch = [0, 0, 0, 0, 0]
    save_checkpoint: bool = True

    save_best_model: bool = True
    log_wandb_project: str = 'my_proiect_1'

    noise_p: float = 0.3
    dropout_p: float = 0.1
    patch_size: int = 256
    y = 2
    b = 1
    log_path = './log_feature/'
    def state_dict(self):
        return {k: getattr(self, k) for k, _ in Path_Hyperparameter.__dict__.items() \
                if not k.startswith('_')}


ph = Path_Hyperparameter()
