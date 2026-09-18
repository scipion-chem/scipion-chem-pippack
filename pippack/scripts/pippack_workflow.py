import os
import sys
import argparse
import pickle
import warnings

import torch
import lightning
import hydra


warnings.filterwarnings("ignore")


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--pippack_path', required=True)
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--model_weights', required=True)

    parser.add_argument(
        '--mse_to_met',
        action='store_true'
    )

    parser.add_argument(
        '--chi_temperature',
        type=float,
        default=1.0
    )

    parser.add_argument(
        '--recycles',
        type=int,
        default=3
    )

    parser.add_argument(
        '--use_resampling',
        action='store_true'
    )

    parser.add_argument(
        '--sample_temp',
        type=float,
        default=0.1
    )

    parser.add_argument(
        '--clash_overlap_tolerance',
        type=float,
        default=0.4
    )

    parser.add_argument(
        '--pro_tolerance_factor',
        type=int,
        default=12
    )

    parser.add_argument(
        '--max_iters',
        type=int,
        default=50
    )

    parser.add_argument(
        '--metropolis_temp',
        type=float,
        default=0.000005
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None
    )

    parser.add_argument(
        '--gpu_ids',
        type=str,
        default='0'
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # ------------------------------------------------------------
    # Use the cloned PIPPack repository directly.
    # ------------------------------------------------------------

    pippack_path = os.path.abspath(args.pippack_path)

    if pippack_path not in sys.path:
        sys.path.insert(0, pippack_path)

    # ------------------------------------------------------------
    # Import PIPPack's own implementation.
    # ------------------------------------------------------------

    from data.protein import from_pdb_file
    from data.top2018_dataset import (
        transform_structure,
        collate_fn
    )
    from inference import pdbs_from_prediction
    from ensembled_inference import sample_epoch
    from model.resampling import resample_loop
    from utils.train_utils import load_checkpoint

    # ------------------------------------------------------------
    # Get number of chi bins from PIPPack's model config.
    # This is exactly what their notebook does.
    # ------------------------------------------------------------

    weights_path = os.path.abspath(args.model_weights)

    model_names = [
        'pippack_model_1',
        'pippack_model_2',
        'pippack_model_3'
    ]

    cfg_file = os.path.join(
        weights_path,
        f'{model_names[0]}_config.pickle'
    )

    with open(cfg_file, 'rb') as f:
        cfg = pickle.load(f)

    n_chi_bins = cfg.model.n_chi_bins

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------
    if (
        args.gpu_ids
        and torch.cuda.is_available()
    ):
        device = torch.device(
            f'cuda:{args.gpu_ids}'
        )
    else:
        device = torch.device('cpu')

    print(
        f'[PIPPack] Device: {device}',
        flush=True
    )

    # ------------------------------------------------------------
    # Load PIPPack models.
    # ------------------------------------------------------------

    models = []

    for model_name in model_names:

        cfg_file = os.path.join(
            weights_path,
            f'{model_name}_config.pickle'
        )

        ckpt_file = os.path.join(
            weights_path,
            f'{model_name}_ckpt.pt'
        )

        with open(cfg_file, 'rb') as f:
            cfg = pickle.load(f)

        model = hydra.utils.instantiate(
            cfg.model
        ).to(device)

        load_checkpoint(
            ckpt_file,
            model
        )

        models.append(model)

    # ------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------

    if 'PL_GLOBAL_SEED' in os.environ:
        os.environ.pop('PL_GLOBAL_SEED')

    lightning.seed_everything(
        args.seed
    )

    # ------------------------------------------------------------
    # Resampling arguments.
    # ------------------------------------------------------------

    resample_args = {
        'sample_temp': args.sample_temp,
        'clash_overlap_tolerance':
            args.clash_overlap_tolerance,
        'pro_tolerance_factor':
            args.pro_tolerance_factor,
        'max_iters': args.max_iters,
        'metropolis_temp':
            args.metropolis_temp,
    }

    # ------------------------------------------------------------
    # Process every PDB supplied by Scipion.
    # ------------------------------------------------------------

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    pdb_files = sorted(
        f
        for f in os.listdir(args.input_dir)
        if f.endswith('.pdb')
    )

    for pdb_file in pdb_files:

        pdb_path = os.path.join(
            args.input_dir,
            pdb_file
        )

        pdb_name = os.path.splitext(
            pdb_file
        )[0]

        print(
            f'[PIPPack] Processing {pdb_file}',
            flush=True
        )

        protein = vars(
            from_pdb_file(
                pdb_path,
                chain_id=None,
                mse_to_met=args.mse_to_met
            )
        )

        protein = transform_structure(
            protein,
            n_chi_bins,
            sc_d_mask_from_seq=True
        )

        batch = collate_fn(
            [protein]
        )

        sample_results = sample_epoch(
            models,
            batch,
            args.chi_temperature,
            device,
            n_recycle=args.recycles
        )

        if args.use_resampling:

            for i in range(
                batch.S.shape[0]
            ):

                temp_protein = {
                    "S":
                        sample_results["S"][i],

                    "X":
                        sample_results["X"][i],

                    "X_mask":
                        sample_results["X_mask"][i],

                    "BB_D":
                        sample_results["BB_D"][i],

                    "residue_index":
                        sample_results["residue_index"][i],

                    "residue_mask":
                        sample_results["residue_mask"][i],

                    "chi_logits":
                        sample_results["chi_logits"][i],

                    "chi_bin_offset":
                        sample_results["chi_bin_offset"][i]
                        if "chi_bin_offset"
                        in sample_results
                        else None,
                }

                pred_xyz = (
                    sample_results["final_X"][i]
                )

                resample_xyz, _ = resample_loop(
                    temp_protein,
                    pred_xyz,
                    **resample_args
                )

                sample_results[
                    "final_X"
                ][i] = resample_xyz

        pdb_str = pdbs_from_prediction(
            sample_results
        )[0]

        output_file = os.path.join(
            args.output_dir,
            f'{pdb_name}_pippack.pdb'
        )

        with open(
            output_file,
            'w'
        ) as f:
            f.write(pdb_str)

        print(
            f'[PIPPack] Written: {output_file}',
            flush=True
        )


if __name__ == '__main__':
    main()