# Paper Title：FGRCAT: A Fine-Grained Reasoning Framework through Causality and Adversarial Training

### Pre-trained language models have revolutionized natural language reasoning (NLR) tasks by learning intricate patterns and relationships from vast amounts of text data. However, existing methods exhibit notable limitations in their ability to memorize and store comprehensive language knowledge, and their understanding of language logic remains constrained. To address these challenges, several innovative works have proposed integrating external knowledge bases and databases to enhance model understanding capabilities. Despite some progress, these methods still fall short in analyzing causalities within abstract natural language in complex scenarios and fail to account for the sensitivity of features among entities in the data, thereby impeding the model's reasoning performance. To bridge this gap, we introduce FGRCAT (Fine-Grained Reasoning with Causality and Adversarial Training), a novel framework designed to enhance the model's reasoning ability by capturing causalities and subtle, sensitive features of data. FGRCAT achieves this by refining causal analysis in natural language through a combination of feature transformation and semantic inversion. This innovative approach ensures that the model comprehensively grasps critical information among entities in the data. We conduct an exhaustive evaluation of FGRCAT on three benchmark datasets, showcasing its effectiveness in NLR tasks. The results demonstrate that FGRCAT outperforms baseline models by an average accuracy improvement of over 10\% and even surpasses many popular large language models in terms of accuracy. This underscores FGRCAT's superiority in capturing complex relationships and nuances in natural language data.



[exampleqa.pdf](https://github.com/user-attachments/files/16599144/exampleqa.pdf)



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

