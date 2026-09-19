import argparse
import yaml
from easydict import EasyDict


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def main():
    with open('configs/train.yaml', 'r') as f:
        train_args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))
    noise_layers = train_args.noise_layers

    defaults = dict(
        image_size=128,
        message_length=30,
        embedding_dim=256,
        num_channels=64,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="16,8",
        channel_mult="",
        conv_resample=True,
        dropout=0.1,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=False,
        pool="linear",
        learn_sigma=False,
        diffusion_steps=100,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=True,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )

    train_defaults = dict(
        data_dir="CelebA-HQ/train_128",
        schedule_sampler="uniform",
        threshold=10000,
        lr=1e-4,
        weight_decay=1e-5,
        lr_anneal_steps=0,
        batch_size=16,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        local_rank=0,
        noise_layers=noise_layers,
    )

    defs = defaults.copy()
    defs.update(train_defaults)

    parser = argparse.ArgumentParser()
    for k, v in defs.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

    args = parser.parse_args([])
    print(args)


if __name__ == "__main__":
    main()
