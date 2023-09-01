# Savanna

Transformer alternatives (pretraining, evals, inference, synthetics). Based on an initial fork of GPT-NeoX.

## Environment Variables

Set `SAVANNA_PATH` to the main folder.

## Common Issues

```
ImportError: cannot import name 'helpers' from 'megatron.data' 
```

`cd ./savanna/data && make` (see: https://github.com/EleutherAI/gpt-neox/issues/934).