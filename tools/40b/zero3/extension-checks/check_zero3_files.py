import argparse
import os

from partition_lib import (
    create_zero3_model_state_path,
    create_zero3_optim_state_path,
    get_model_files_by_mp_rank,
    get_model_mp_dp_ranks,
    get_optim_files_by_mp_rank,
    get_optim_mp_dp_ranks,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=str)
    parser.add_argument("--mp_rank", type=int, required=True)
    parser.add_argument("--dp_rank", type=int, required=True)
    args = parser.parse_args()

    expected_model_files = [create_zero3_model_state_path(dp_rank, mp_rank) for mp_rank in range(args.mp_rank) for dp_rank in range(args.dp_rank)]
    expected_optim_files = [create_zero3_optim_state_path(dp_rank, mp_rank) for mp_rank in range(args.mp_rank) for dp_rank in range(args.dp_rank)]

    model_files = []
    optim_files = []
    for mp_rank in range(args.mp_rank):
        model_files.extend([os.path.basename(f) for f in get_model_files_by_mp_rank(args.checkpoint_dir, mp_rank)])
        optim_files.extend([os.path.basename(f) for f in get_optim_files_by_mp_rank(args.checkpoint_dir, mp_rank)])

    print(f"Model files: {len(model_files)}")
    print(f"Optim files: {len(optim_files)}")

    missing_model_files = [f for f in expected_model_files if f not in model_files]
    missing_optim_files = [f for f in expected_optim_files if f not in optim_files]

    if missing_model_files or missing_optim_files:
        if missing_model_files:
            print(f"Missing {len(missing_model_files)} model files:")
            for f in missing_model_files:
                dp, mp = get_model_mp_dp_ranks(f)
                print(f"  {f} (dp={dp}, mp={mp})")
        if missing_optim_files:
            print(f"Missing {len(missing_optim_files)} optim files: {missing_optim_files}")
            for f in missing_optim_files:
                dp, mp = get_optim_mp_dp_ranks(f)
                print(f"  {f} (dp={dp}, mp={mp})")

    else:
        print("All files present")
