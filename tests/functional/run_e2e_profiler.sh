MODULE_NAME=savanna
MODULE_ROOT=$(python -c "
import os, importlib.util
spec = importlib.util.find_spec('$MODULE_NAME')
if spec:
    module_dir = os.path.dirname(spec.origin)
    print(os.path.abspath(os.path.join(module_dir, os.pardir)))
else:
    print('Module not found')
")

# Check if the module root was found
if [ "$MODULE_ROOT" == "Module not found" ]; then
    echo "Module $MODULE_NAME not found."
    exit 1
fi

# Print or use the module root directory
cd $MODULE_ROOT
CONFIGS=(tests/test_configs/profiler/e2e/*.yml)
for config in "${CONFIGS[@]}"; do
    CMD="python tools/slurm_runner.py --nodes=1 --gpus-per-node=4 --model-config $config"

    case "$config" in
  *"nsys"*)
    CMD="$CMD --nsys"
    ;;
  *)
    CMD="$CMD"
    ;;
esac
    echo "Running command: $CMD"
    $CMD
done