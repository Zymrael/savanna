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

`implicit_real_modal`: Real modal diagonal parametrization (poles and residues), uses a mSISO (multi-input single-output) structure (each channel has multiple states and thus channels)

`implicit_complex_modal`: Complex modal diagonal baseline parametrization (poles + residues)


## Testing

We adopt a simple testing regression testing protocol. Every PR should run the following configs:

```
python launch.py train.py -d configs data/opengenome.yml configs/test/regression_1.yml
python launch.py train.py -d configs data/opengenome.yml configs/test/regression_2.yml
```

where:

- `regression_1.yml`: tests a generic Hyena cascade model 
- `regression_2.yml`: tests a generic Hyena cascade model with tensor parallelism

we use `opengenome` as the testing dataset as it is faster to load.


## Custom Kernels

`gcg_two_pass`: Chunked convolutions via two-pass GEMMs. Can be used with any finite impulse response convolution (windowed) parametrization (e.g., `explicit` or `explicit_single_decay`)

`hyena_

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


```
https://github.com/HazyResearch/safari-neox/blob/d6c7dd959e52d90396261bbca9e393bc1db54128/savanna/training.py#L741
I think this might explain the difference between pipeline size 0 and 1. 
0 did not do grad accumulation properly. We should put optimizer step outside the for loop.
```

Disabling `pipeline` results in an issue when gradient accumulation steps `> 1`

## Style

We use `black` (`pip install black`) for code formatting:

```
black -x --safe --line-length=110 .
```
