# Paper Title：FGRCAT: A Fine-Grained Reasoning Framework through Causality and Adversarial Training


## python = 3.6
## torch = 2.1.2
## transformers = 4.30.2
## scikit-learn = 1.30.2



  ```shell
  python3 train_discriminate.py \
    --data_dir "../dataset/Causal_Reasoning/" \
    --model_dir "../../huggingface_transformers/bart-base/" \
    --save_dir "./output/saved_model" \
    --log_dir "./output/log" \
    --train "train.jsonl" \
    --dev "dev.jsonl" \
    --test "dev.jsonl" \
    --model_name "bart" \
    --gpu "0" \
    --batch_size 64 \
    --cuda True\
    --epochs 100 \
    --evaluation_step 200 \
    --lr 1e-5 \
    --set_seed True \
    --seed 338 \
    --patient 3 \
    --loss_func "BCE" \
  ```

