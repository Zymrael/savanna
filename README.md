<p align="center">
    <h1>Savanna</h1>
</p>

Infrastructure dedicated pretraining and model research around application, scaling and parametrization of deep signal processing architectures (Hyena, HyenaDNA, Evo...)

<p align="center">
  <img src="https://github.com/user-attachments/assets/be1b2a06-da7e-4de7-b8a7-a4b78ea42ab5", width="260">
</p>

## Environment Variables

Set `SAVANNA_PATH` to the main folder.

## Quick Start Setup

```bash
python launch.py train.py -d configs data/opengenome.yml model/evo1/8e18_hyena_134m_10k.yml
```

## Model 

### Filter Cascade

Savanna is designed to train flexible hybrid (striped) architectures, composed by a mixture of modern gated convolutions (hyena) and attention operators. We provide a set of parametrization for convolutional filters, including explicit and implicit variants. 

#### Parametrizations

`implicit_freeform`: Free-form parametrization of the filters via a shallow network with sine activations, followed by exponential decays of different rates.

`explicit_single_decay`: Explicit parametrization of filter coefficients in time domain, followed by exponential decays.

`implicit_modal`: Real modal diagonal parametrization (poles and residues), uses a mSISO (multi-input single-output) structure

`implicit_complex_modal`: Complex modal diagonal baseline parametrization (poles + residues)


## Custom Kernels

`gcg_two_pass`: Chunked convolutions via two-pass GEMMs. Can be used with any finite impulse response convolution (windowed) parametrization (e.g., `explicit` or `explicit_single_decay`)

`hyena_short`: custom kernel for short Hyena layers

## Infrastructure 

Savanna supports lower precision pretraining (bf16, fp32), including fp8 on dense linear layers. 

To activate fp8, set `use_fp8_linears` to `true` in model configs file. 

### Checkpoints

#### GCP

Some checkpoints are kept on GCP and in R2 buckets (AWS). To install `gcloud` on `Ubuntu 21.10+` (follow [this](https://cloud.google.com/sdk/docs/install) if the commands below do not work). 

```
apt-get update && \
apt-get install -y apt-transport-https ca-certificates gnupg curl sudo && \
echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
apt-get update && apt-get install google-cloud-cli
```

#### Local storage

Most model checkpoints are also available locally (e.g., on the Arc cluster). 

### Known Issues

```
ImportError: cannot import name 'helpers' from 'megatron.data' 
```

`cd ./savanna/data && make`

## Style

We use `black` (`pip install black`) for code formatting:

```
black -x --safe --line-length=110 .
```

## Testing

### Regression Testing
When submitting PRs, please make sure to run the regression test SLURM job `./run_regression_test.sh`

The script runs two pre-defined model configs, a single-gpu and 4-gpu model parallel config.  The script automatically allocates the necessary resources via SLURM.
- Usage:
  - Run the script from within the savanna root directory. 
  - Alternatively, provide the required paths to the data and model configs.
  - See `./run_regression_test.sh --help` to see full list of commandline options (config paths, checkpoint, log directories, etc.)
- Please run with the default number of iterations (2000) if making model architecture or parallelism changes.  
- For minor infrastructure changes, ok to run -- to completion -- with a smaller number of iterations
  - E.g. `./run_regression_test.sh --train-iters-1 100 --train-iters-2 100`

### Code Style
Install `ruff`, an all-in-one code linting and formatting tool:
```
pip install ruff
```
Then run:
```
make check
```
This will call ruff to check all files in `savanna` and enumerate formatting / linting fixes without actually making any code changes.

Run:
```
make style
```
to then fix these changes.  Not all changes will be fixable, some will need be manually fixed.

