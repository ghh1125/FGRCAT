import json

def process_jsonl(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line.strip())
            cause_words = data['cause'].split()
            effect_words = data['effect'].split()
            conceptual_explanation = ' '.join([word for word in effect_words if word not in cause_words])
            data['conceptual_explanation'] = conceptual_explanation
            json.dump(data, f_out)
            f_out.write('\n')

if __name__ == "__main__":
    process_jsonl("trainEG.jsonl", "processed_trainEG.jsonl")
