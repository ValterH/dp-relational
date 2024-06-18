PYTHON_SCRIPT="experiment_temp.py"
LOG_FILE="script.log"
bash -c "python3 $PYTHON_SCRIPT > $OUTPUT_LOG 2>&1; echo 'Python script completed.'; exec bash"

