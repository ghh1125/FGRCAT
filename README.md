# Paper Title：FGRCAT: A Fine-Grained Reasoning Framework through Causality and Adversarial Training

##### Pre-trained language models have revolutionized natural language reasoning (NLR) tasks by learning intricate patterns and relationships from vast amounts of text data. However, existing methods exhibit notable limitations in their ability to memorize and store comprehensive language knowledge, and their understanding of language logic remains constrained. To address these challenges, several innovative works have proposed integrating external knowledge bases and databases to enhance model understanding capabilities. Despite some progress, these methods still fall short in analyzing causalities within abstract natural language in complex scenarios and fail to account for the sensitivity of features among entities in the data, thereby impeding the model's reasoning performance.


![image](https://github.com/user-attachments/assets/edaa92f0-2ecf-4feb-8d86-159d3cf4df49)

## Our method

![image](https://github.com/user-attachments/assets/506aa11a-0384-4a68-84e7-238ece49e1c1)


### Causality Enhancement Model Reasoning

### Adversarial Training Enhancement Model Reasoning




## Run

```bash
conda create -n FGRCAT python=3.9
conda activate FGRCAT
pip install -r requirements.txt
```




  ```shell
  python3 train_discriminate.py \   # or you can choose QAtrain.py (QA datasets)
    --data_dir "../dataset/" \    # choose datasets
    --model_dir "../../huggingface_transformers/bart-base/" \       # choose model(huggingface)
    --save_dir "./output/saved_model" \
    --log_dir "./output/log" \
    --train "train.jsonl" \
    --dev "dev.jsonl" \
    --test "dev.jsonl" \
    --model_name "bart" \
    --gpu "0" \
    --batch_size 64 \
    --cuda True\
    --epochs 50 \
    --evaluation_step 200 \
    --lr 1e-5 \
    --set_seed True \
    --seed 338 \
    --patient 3 \
    --loss_func "BCE" \  # choose loss function
  ```

