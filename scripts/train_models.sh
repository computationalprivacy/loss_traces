exp_id="test"
cpus="19-20"
gpu=":0"

dataset="CIFAR10"
arch="wrn28-2"
epochs=1
lr=0.05
bs=64

shadow_count=2
dual_count=1


# Train 1st model 
echo "Training target model"
start_time=$(date +%s)
taskset -c $cpus python $norm_tracker_path/main.py --track_grad_norms --gpu $gpu --dataset $dataset --seed 34568 --arch $arch --batchsize $bs --lr $lr --epochs $epochs --exp_id $exp_id
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Took $elapsed_time seconds"

## Train 'dual' models on same training set, varying randomness for batch selection
for ((i=0; i<$dual_count; i++))
do
    echo "Training dual model $i"
    start_time=$(date +%s) #
    taskset -c $cpus python $norm_tracker_path/main.py --track_grad_norms --gpu $gpu --dataset $dataset --seed 8574${i} --dual track_both_$i --arch $arch --batchsize $bs --lr $lr --epochs $epochs --exp_id $exp_id
    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    echo "Took $elapsed_time seconds"
done

# Train shadow models
echo "Training shadow models"
start_time=$(date +%s)
taskset -c $cpus python $norm_tracker_path/main.py --model_start 0 --model_stop $shadow_count --gpu $gpu --dataset $dataset --arch $arch --batchsize $bs --lr $lr --epochs $epochs --shadow_count $shadow_count --exp_id $exp_id
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Took $elapsed_time seconds"

## Compute LiRA for each target/dual model
echo "Starting lira"
start_time=$(date +%s)
taskset -c $cpus python $norm_tracker_path/attacks/lira.py --exp_id $exp_id --gpu $gpu
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Took $elapsed_time seconds"
for ((i=0; i<$dual_count; i++))
do
    echo "Starting lira dual ${i}"
    taskset -c $cpus python $norm_tracker_path/attacks/lira.py --exp_id $exp_id --gpu $gpu --target_id dual_track_both_$i
done
