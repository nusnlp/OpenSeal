import argparse
import json
import os
import sys
import re


regex_string = {
    "id": r"(?<=[T|t]erjemahan:)[\s\r\n]*.*",
    "th": r"(?<=คำแปล:)[\s\r\n]*.*",
    "vi": r"(?<=[B|b]ản dịch:)[\s\r\n]*.*",
    "ta": r"(?<=மொழிபெயர்ப்பு:)[\s\r\n]*.*",
    "jv": r"(?<=[T|t]erjemahan:)[\s\r\n]*.*",
    "su": r"(?<=[T|t]arjamahan:)[\s\r\n]*.*",
    "tl": r"(?<=[S|s]alin:)[\s\r\n]*.*",
    "km": r"(?<=ការបកប្រែ៖)[\s\r\n]*.*",
    "lo": r"(?<=ການແປ:)[\s\r\n]*.*",
    "my": r"(?<=ဘာသာပြန်ချက်-)[\s\r\n]*.*",
    "ms": r"(?<=[T|t]erjemahan:)[\s\r\n]*.*",
    "zh": r"(?<=翻译：)[\s\r\n]*.*"
}


def parse_response(response: list, lang: str, use_lowercase: bool = False) -> str:
        # try to extract the answer from the response using regex else return the response as it is
        if use_lowercase:
            _response = response[0].lower()
        else:
            _response = response[0]

        try:
            output = re.search(regex_string[lang], _response).group(0)
            output = output.strip("$")
        except:
            output = _response

        return output.strip()


def extract_responses(input_path: str, output_path: str, lang: str = None) -> int:
    """
    Read a JSONL file (one JSON object per line) from input_path.
    Write the value of the "responses" key from each JSON object as a JSON-encoded
    line into output_path.
    Returns number of successfully written lines.
    """
    written = 0
    # ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for lineno, raw in enumerate(infile, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"Line {lineno}: JSON decode error: {e}\n")
                continue
            if "responses" not in obj:
                sys.stderr.write(f"Line {lineno}: missing 'responses' key, skipping\n")
                continue
            # write the responses value as a JSON-encoded line
            
            # outfile.write(json.dumps(parse_response(obj["responses"], lang="en"), ensure_ascii=False) + "\n")
            clean_response = obj.get("cleaned_response", None) or parse_response(obj.get("responses", ""), lang=lang)
            outfile.write(clean_response.strip().replace("\n", " ") + "\n")
            written += 1
    return written


def parse_args():
    parser = argparse.ArgumentParser(description="Extract 'responses' from each JSON object in a JSONL file.")
    parser.add_argument("input", help="Path to input JSONL file (one JSON object per line).")
    parser.add_argument("--output", default=None, help="Path to output file where extracted responses will be written (one JSON value per line).")
    parser.add_argument("--lang", default=None, help="Language code for regex extraction.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output is None:
        input_no_ext = os.path.splitext(args.input)[0]
        output_path = input_no_ext + ".responses.txt"
    else:
        output_path = args.output
    count = extract_responses(args.input, output_path, args.lang)
    print(f"Wrote {count} lines to {output_path}")

if __name__ == "__main__":
    main()
