#!/bin/bash
#SBATCH --job-name=translation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

# python3 -m venv ../env_seahelm
source ../env_seahelm/bin/activate
# git clone https://github.com/ThuraAung1601/myTokenize.git
# cd myTokenize
# pip install -r requirements.txt 
# pip install .
# cd ..
# pip install khmer-nltk laonlp indic-nlp-library pythainlp

# Add a list of models (either local path or HuggingFace model id) to be evaluated
# MODEL="aisingapore/Qwen-SEA-LION-v4-32B-IT"
MODEL=${1:-/home/llm3/openseal-project/models/OpenSeal/7B/para-only-7B}

PARENT="../outputs/translation-results"
INPUT_DIR="${PARENT}/$(echo ${MODEL} | awk -F/ '{print $(NF-1)}')"
INPUT_PATH="${INPUT_DIR}/$(basename $MODEL)/inference"
OUTPUT_DIR="${INPUT_DIR}/$(basename $MODEL)/inference_texts"
mkdir -p "$OUTPUT_DIR"

JSONL_FILES=("$INPUT_PATH"/*.jsonl)

export SACREBLEU_FORMAT=text

for JSONL_FILE in "${JSONL_FILES[@]}"; do
	echo "processing $JSONL_FILE"
        OUTPUT_FILE="$OUTPUT_DIR/$(basename "${JSONL_FILE%.jsonl}_parsed.txt")"

        lang="en"
        # check if file contains "xx-en" to set lang="en"
        if [[ "$JSONL_FILE" == *"xx-en"* ]]; then
            lang="en"
        fi

        # sample: Llama-SEA-LION-v3.5-8B-R_translation-en-xx_id.jsonl
        # then language is "id"
        if [[ "$JSONL_FILE" == *"en-xx"* ]]; then
            lang=$(echo "$JSONL_FILE" | sed -n 's/.*en-xx_\([a-z]*\)\.jsonl/\1/p')
        fi

        # support tagalog (tl) file
        # sample: Gemma-SEA-LION-v4-27B_translation-en-xx_tl_full.jsonl
        if [[ "$JSONL_FILE" == *"en-xx"* ]]; then
            lang=$(echo "$JSONL_FILE" | sed -n 's/.*en-xx_\([a-z]*\)\(_full\)\?\.jsonl/\1/p')
        fi

        # Validate variables before processing
        if [[ -z "$JSONL_FILE" ]] || [[ ! -f "$JSONL_FILE" ]]; then
            echo "WARNING: Skipping invalid file: $JSONL_FILE"
            continue
        fi

        echo "Processing file: $JSONL_FILE for language: $lang"
        echo "Output: $OUTPUT_FILE"

        # Direct Python call
        # python translation-scripts/seahelm_tasks/nlg/translation/parse_json.py \
        python3 seahelm_tasks/nlg/translation/parse_json.py \
            "$JSONL_FILE" \
            --output "$OUTPUT_FILE" \
            --lang "$lang"

        if [ $? -eq 0 ]; then
            echo "✓ Successfully processed: $JSONL_FILE"
        else
            echo "✗ Error processing: $JSONL_FILE"
        fi

        echo "------------------------------------------"

	if [[ ! -f "$OUTPUT_FILE" ]]; then
            echo "No parsed text files found in $OUTPUT_FILE"
            continue
        fi

        # sample: Llama-SEA-LION-v3.5-8B-R_translation-en-xx_id_parsed.txt
        # then language is "id"
        if [[ "$OUTPUT_FILE" == *"-en-xx_"* ]]; then
            lang=$(echo "$OUTPUT_FILE" | sed -n 's/.*-en-xx_\([a-z]*\)_parsed\.txt/\1/p')

            # only apply for these languages: km, lo, my, ta, th
            if [[ ! " km lo my ta th " =~ " $lang " ]]; then
                echo "Skipping file: $OUTPUT_FILE for unsupported language: $lang"
                echo "------------------------------------------"
                continue
            fi

            echo "Tokenizing file: $OUTPUT_FILE for language: $lang"

            # python translation-scripts/seahelm_tasks/nlg/translation/tokenize_text.py \
            python3 seahelm_tasks/nlg/translation/tokenize_text.py \
                "$OUTPUT_FILE" \
                "$lang"

            if [ $? -eq 0 ]; then
                echo "✓ Successfully tokenized: $OUTPUT_FILE"
            else
                echo "✗ Error tokenizing: $OUTPUT_FILE"
            fi

            echo "------------------------------------------"
        fi
    done

bash score_translation.sh $OUTPUT_DIR
