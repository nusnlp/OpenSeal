import argparse
import io
import sys
import re
from typing import List
import time, signal

TIMEOUT_SEC = 1

class TimeoutError_(Exception):
    pass

def _alarm_handler(signum, frame):
    raise TimeoutError_()

signal.signal(signal.SIGALRM, _alarm_handler)

def tokenize_line(line: str, lang: str) -> List[str]:
    lang = lang.lower()
    if lang in ("th", "thai"):
        try:
            from pythainlp.tokenize import word_tokenize as thai_word_tokenize
        except ImportError:
            raise RuntimeError("pythainlp is required for Thai tokenization. Install with: pip install pythainlp")
        # use NewMM engine
        return thai_word_tokenize(line, engine="newmm")
    # elif lang in ("vi", "vietnamese"):
    #     try:
    #         from pyvi import ViTokenizer
    #     except ImportError:
    #         raise RuntimeError("pyvi is required for Vietnamese tokenization. Install with: pip install pyvi")
    #     tok = ViTokenizer.tokenize(line)
    #     return tok.split()
    elif lang in ("ta", "tamil"):
        try:
            from indicnlp.tokenize import indic_tokenize
        except ImportError:
            raise RuntimeError("indic-nlp-library is required for Tamil tokenization. Install with: pip install indic-nlp-library")
        return indic_tokenize.trivial_tokenize(line, lang="ta")
    elif lang in ("km", "khmer", "cambodian"):
        try:
            from khmernltk import word_tokenize as khmer_word_tokenizer
        except ImportError:
            raise RuntimeError("khmernltk is required for Khmer tokenization. Install with: pip install khmer-nltk")
        return khmer_word_tokenizer(line, return_tokens=True)
    elif lang in ("lo", "lao"):
        try:
            from laonlp.tokenize import word_tokenize as lo_word_tokenize
        except ImportError:
            raise RuntimeError("laonlp is required for Lao tokenization. Install with: pip install laonlp")
        
        def safe_tokenize(text, timeout=TIMEOUT_SEC):
            # Enforce a hard timeout on word_tokenize
            signal.alarm(timeout)
            try:
                tokens = lo_word_tokenize(line)
                return tokens
            except TimeoutError_:
                return line
            except Exception as e:
                return line
            finally:
                signal.alarm(0)  # always clear
        
        return safe_tokenize(line)
    elif lang in ("my", "myanmar", "burmese"):
        try:
            from myTokenize import WordTokenizer as myWordTokenizer
        except ImportError:
            raise RuntimeError("myTokenize is required for Myanmar tokenization. Install with: pip install myTokenize")
        my_word_tokenizer = myWordTokenizer()
        return my_word_tokenizer.tokenize(line)
    else:
        raise NotImplementedError(f"Tokenization for language '{lang}' is not implemented.")


def main(args):

    out_path = args.output if args.output else args.input_file + ".tok"

    try:
        with io.open(args.input_file, "r", encoding="utf-8") as inf, io.open(out_path, "w", encoding="utf-8") as outf:
            for line in inf:
                stripped = line.rstrip("\n")
                if not stripped:
                    outf.write("\n")
                    continue
                try:
                    tokens = tokenize_line(stripped, args.language)
                except RuntimeError as e:
                    print("Error:", e, file=sys.stderr)
                    sys.exit(2)
                outf.write(" ".join(tokens) + "\n")
    except FileNotFoundError:
        print(f"Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Tokenized file written to: {out_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Tokenize a text file by language (Thai->NewMM, Vietnamese->Pyvi).")
    parser.add_argument("input_file", help="Path to input text file (utf-8).")
    parser.add_argument("language", help="Language code/name (e.g., 'th'|'thai' or 'vi'|'vietnamese').")
    parser.add_argument("-o", "--output", help="Output file path. Defaults to input_file + '.tok'")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
