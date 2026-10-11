
INPUT_PATH=$1

echo "Scoring translation for $INPUT_PATH"

ENG_FILES=("$INPUT_PATH"/*-xx-en_*_parsed.txt)
ENG_REF_FILE="data/translation/ref/en.devtest"
for ENG_FILE in "${ENG_FILES[@]}"; do
        BASE_NAME=$(basename "$ENG_FILE")

        OUTPUT_DIR="$INPUT_PATH/bleu_results"
        mkdir -p "$OUTPUT_DIR"
        OUTPUT_FILE="$OUTPUT_DIR/$BASE_NAME.bleu"

        echo "Running: sacrebleu $ENG_REF_FILE -w 2 -i $ENG_FILE"

        sacrebleu "$ENG_REF_FILE" -w 2 -i "$ENG_FILE" 2>&1 | tee "$OUTPUT_FILE"

        if [ $? -eq 0 ]; then
            echo "✓ Successfully calculated BLEU for: $ENG_FILE"
        else
            echo "✗ Error calculating BLEU for: $ENG_FILE"
        fi
    done

    SEA_FILES=("$INPUT_PATH"/*-en-xx_*_parsed.txt)
    for SEA_FILE in "${SEA_FILES[@]}"; do
        BASE_NAME=$(basename "$SEA_FILE")
        LANG_CODE=$(echo "$BASE_NAME" | sed -n 's/.*-en-xx_\([a-z][a-z]\)_parsed.txt/\1/p')

        # Reset variables for each iteration
        TOK=""
        REF_FILE=""
        CURRENT_SEA_FILE="$SEA_FILE"

        if [[ "$LANG_CODE" == "id" ]]; then
            REF_FILE="data/translation/ref/id.devtest"
            TOK="intl"
        elif [[ "$LANG_CODE" == "km" ]]; then
            REF_FILE="data/translation/ref/km.devtest"
            CURRENT_SEA_FILE="${SEA_FILE}.tok"
            TOK="none"
        elif [[ "$LANG_CODE" == "lo" ]]; then
            REF_FILE="data/translation/ref/lo.devtest"
            CURRENT_SEA_FILE="${SEA_FILE}.tok"
            TOK="none"
        elif [[ "$LANG_CODE" == "ms" ]]; then
            TOK="intl"
            REF_FILE="data/translation/ref/ms.devtest"
        elif [[ "$LANG_CODE" == "my" ]]; then
            REF_FILE="data/translation/ref/my.devtest"
            CURRENT_SEA_FILE="${SEA_FILE}.tok"
            TOK="none"
        elif [[ "$LANG_CODE" == "ta" ]]; then
            REF_FILE="data/translation/ref/ta.devtest"
            CURRENT_SEA_FILE="${SEA_FILE}.tok"
            TOK="none"
        elif [[ "$LANG_CODE" == "th" ]]; then
            REF_FILE="data/translation/ref/th.devtest"
            CURRENT_SEA_FILE="${SEA_FILE}.tok"
            TOK="none"
        elif [[ "$LANG_CODE" == "tl" ]]; then
            TOK="intl"
            REF_FILE="data/translation/ref/tl.devtest"
        elif [[ "$LANG_CODE" == "vi" ]]; then
            TOK="intl"
            REF_FILE="data/translation/ref/vi.devtest"
        elif [[ "$LANG_CODE" == "zh" ]]; then
            REF_FILE="data/translation/ref/zh.devtest"
            TOK="zh"
        else
            echo "Unknown language code: $LANG_CODE for file: $SEA_FILE"
            continue
        fi

        OUTPUT_DIR="$INPUT_PATH/bleu_results"
        mkdir -p "$OUTPUT_DIR"
        OUTPUT_FILE="$OUTPUT_DIR/$BASE_NAME.bleu"

        TOK_ARG=""
        if [ -n "$TOK" ]; then
            TOK_ARG="-tok $TOK"
        fi

        echo "Running: sacrebleu $REF_FILE -w 2 -i $CURRENT_SEA_FILE $TOK_ARG"

        sacrebleu "$REF_FILE" -w 2 -i "$CURRENT_SEA_FILE" $TOK_ARG 2>&1 | tee "$OUTPUT_FILE"

        if [ $? -eq 0 ]; then
            echo "✓ Successfully calculated BLEU for: $CURRENT_SEA_FILE"
        else
            echo "✗ Error calculating BLEU for: $CURRENT_SEA_FILE"
        fi
    done

    TAGALOG_FILES=("$INPUT_PATH"/*-en-xx_tl_full_parsed.txt)
    for TAGALOG_FILE in "${TAGALOG_FILES[@]}"; do
        BASE_NAME=$(basename "$TAGALOG_FILE")
        LANG_CODE="tl"
        REF_FILE="data/translation/ref/tl.devtest"
        TOK="intl"
        CURRENT_TAGALOG_FILE="$TAGALOG_FILE"

        OUTPUT_DIR="$INPUT_PATH/bleu_results"
        mkdir -p "$OUTPUT_DIR"
        OUTPUT_FILE="$OUTPUT_DIR/$BASE_NAME.bleu"

        TOK_ARG=""
        if [ -n "$TOK" ] && [ "$TOK" != "none" ]; then
            TOK_ARG="-tok $TOK"
        fi

        echo "Running: sacrebleu $REF_FILE -w 2 -i $CURRENT_TAGALOG_FILE $TOK_ARG"

        sacrebleu "$REF_FILE" -w 2 -i "$CURRENT_TAGALOG_FILE" $TOK_ARG 2>&1 | tee "$OUTPUT_FILE"

        if [ $? -eq 0 ]; then
            echo "✓ Successfully calculated BLEU for: $CURRENT_TAGALOG_FILE"
        else
            echo "✗ Error calculating BLEU for: $CURRENT_TAGALOG_FILE"
        fi
    done
