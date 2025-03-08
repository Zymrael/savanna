# Runs tests and parses diffs so that they can be analyzed in the accompanying notebook
TEST_TO_RUN=$1
TEST_NAME="${TEST_TO_RUN%.*}"
LOG_DIR="logs"
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR" 
fi

if [ -z "$TEST_TO_RUN" ]; then
    echo "Usage: $0 <test_file>"
    exit 1
fi

DEBUG_STR="__DEBUG__"
LOG_FILE=$LOG_DIR/$TEST_NAME.log
pytest -s --tb=short --cache-clear $TEST_TO_RUN 2>&1 | tee $LOG_FILE && grep -w $DEBUG_STR $LOG_FILE 2>&1 | tee parsed_$LOG_FILE

