command = "CUDA_VISIBLE_DEVICES=0 python main.py --lr 0.01 --train_batch 8192  --epoch 200 --dataset movie --model hshcl --loss_type 4 --cl_rate 0.01 --save_model 0 --N 8 --decay 1e-5"


command = "CUDA_VISIBLE_DEVICES=0 python main.py --lr 0.001 --epoch 200 --dataset gowalla --model hshcl --loss_type 4 --cl_rate 0.05 --save_model 0 --N 16 --decay 1e-5"


command = "CUDA_VISIBLE_DEVICES=0 python main.py --lr 0.001 --epoch 200 --dataset pinterest --model hshcl --loss_type 4 --cl_rate 0.01 --save_model 0 --N 8"


command = "CUDA_VISIBLE_DEVICES=0 python main.py --lr 0.001 --epoch 200 --dataset yelp --model hshcl --loss_type 4 --cl_rate 0.01 --save_model 0 --N 4"


command = "CUDA_VISIBLE_DEVICES=0 python main.py --lr 0.001 --train_batch 8192  --epoch 100 --dataset book --model hshcl --loss_type 4 --cl_rate 0.01 --temp 0.1 --save_model 0 --N 8 --decay 1e-5"


command = "CUDA_VISIBLE_DEVICES=0 python main.py --lr 0.0001 --epoch 5 --dataset dianping --model hshcl --loss_type 4 --cl_rate 0.005 --save_model 0 --N 4 --decay 1e-5"
