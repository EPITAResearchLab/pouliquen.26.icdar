# run for each fold
for i in {0..4}
do
    python main.py $i
    python evaluate_full_val_test.py $i
done