for seed in {1..40}
do
	python train_data_collection.py --save-models 0 --log-tensorboard 0 --seed $seed
done