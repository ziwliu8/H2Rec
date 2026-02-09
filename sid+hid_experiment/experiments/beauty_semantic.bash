## SASRec with Semantic ID
gpu_id=0
dataset="beauty"
seed_list=(42)
ts_user=9
ts_item=4

# 使用semantic ID的SASRec_seq模型
model_name="SASRec_seq"
for seed in ${seed_list[@]}
do
        python main.py --dataset ${dataset} \
                --model_name ${model_name} \
                --hidden_size 64 \
                --train_batch_size 128 \
                --max_len 200 \
                --gpu_id ${gpu_id} \
                --num_workers 8 \
                --num_train_epochs 200 \
                --seed ${seed} \
                --check_path "semantic" \
                --patience 20 \
                --ts_user ${ts_user} \
                --ts_item ${ts_item} \
                --freeze \
                --log \
                --use_semantic_id \
                --semantic_code_file "beauty.code.fixed_rq.12_256.pca128.json"
done

# 也可以直接使用SemanticSASRec_seq模型名
# model_name="SemanticSASRec_seq"
# for seed in ${seed_list[@]}
# do
#         python main.py --dataset ${dataset} \
#                 --model_name ${model_name} \
#                 --hidden_size 64 \
#                 --train_batch_size 128 \
#                 --max_len 200 \
#                 --gpu_id ${gpu_id} \
#                 --num_workers 8 \
#                 --num_train_epochs 200 \
#                 --seed ${seed} \
#                 --check_path "semantic_direct" \
#                 --patience 20 \
#                 --ts_user ${ts_user} \
#                 --ts_item ${ts_item} \
#                 --freeze \
#                 --log \
#                 --semantic_code_file "beauty.code.fixed_rq.12_256.pca128.json"
# done
