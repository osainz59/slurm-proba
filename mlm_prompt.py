import sys
from transformers import pipeline
import warnings
from argparse import ArgumentParser

parser = ArgumentParser('MLM promting')

parser.add_argument('--topk', type=int, default=5, help="How many predictions to display.")

args = parser.parse_args()

nlp = pipeline('fill-mask', model='ixa-ehu/ixambert-base-cased')

input_prompt = sys.stdin.read().strip()
if not input_prompt:
    warnings.warn("Programari ez zaio prompting bidali. Defektuz hurrengo prompt-a erabiliko da: 'Niri sagarrak gustatzen [MASK] !'")
    input_prompt = "Niri sagarrak gustatzen [MASK] !"

def print_results(results):
    token_max_length = max([len(elem['token_str']) for elem in results] + [len("Token")])
    sequence_max_length = max([len(elem['sequence']) for elem in results] + [len("Sequence")])
    header = "| Probability" + " | " + "Token" + " "*(token_max_length - len("Token")) + " | " + "Sequence" + " "*(sequence_max_length-len("Sequence")) + " |"
    print("+" + "-"*(len(header) - 2) + "+")
    print(header)
    print("|" + "-"*(len(header) - 2) + "|")
    for elem in results:
        prob = f"{elem['score']*100:.2f}%"
        prob += " "*(len("Probability") - len(prob))
        token = elem['token_str']
        token += " "*(token_max_length - len(token))
        sequence = elem['sequence']
        sequence += " "*(sequence_max_length - len(sequence))
        print(f"| {prob} | {token} | {sequence} |")
    print("+" + "-"*(len(header) - 2) + "+")

print("Sarrera prompt-a:", input_prompt)
print_results(nlp(input_prompt, top_k=args.topk))
