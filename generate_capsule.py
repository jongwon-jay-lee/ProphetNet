import re
from argparse import ArgumentParser
from bs4 import BeautifulSoup
from tqdm import tqdm

SRC_FORMAT = "role: user [SEP] da: {intent} [SEP] history: [SEP] slots: {slots} [SEP]"


def extract_tag(annotated_utt):
    soup = BeautifulSoup(annotated_utt, features="html.parser")
    plain_text = soup.text
    assert plain_text.find("<") == -1, f"PLAIN_TEXT must not have < or > : {plain_text}"

    s_e_tags = re.findall(r"<[^>]+>", annotated_utt)
    assert len(s_e_tags) % 2 == 0, f"# TAG should be even : {len(s_e_tags)}"
    tag_set = set([])
    for idx in range(0, len(s_e_tags), 2):
        tag_set.add(s_e_tags[idx])

    s_tag_w_values = []
    for tag in tag_set:
        start_tag = tag
        end_tag = "</" + tag[1:]

        curr_pattern = re.compile(f"({start_tag}(.*?){end_tag})")
        tag_w_values = re.findall(curr_pattern, annotated_utt)

        for tag_w_value in tag_w_values:
            curr_value = tag_w_value[-1]
            org_domain_slot_type = tag[1:-1].split(".")
            assert len(org_domain_slot_type) == 3, f"{curr_value} should consist of 3 parts"
            domain_slot_type = "-".join(org_domain_slot_type[1:])
            s_tag_w_values.append(f"{domain_slot_type}-{curr_value}".lower())

    return s_tag_w_values, plain_text


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./19x_capsules.txt")
    parser.add_argument("--src_path", type=str, default="./src_19x_capsules.txt")
    parser.add_argument("--tgt_path", type=str, default="./tgt_19x_capsules.txt")
    args = parser.parse_args()

    intent_list = []
    slot_type_value_list = []
    plain_text_list = []
    total_lines = sum(1 for _ in open(args.input_path, "r", encoding="utf-8"))
    with open(args.input_path, "r", encoding="utf-8") as f_in:
        for line_idx, line in enumerate(tqdm(f_in, total=total_lines, desc="line")):
            line = line.strip()
            tab_split = line.split("\t")
            assert len(tab_split) == 3, f"{line_idx} ERROR - #TAB{len(tab_split)}"
            intent = tab_split[1]
            intent_tab_split = intent.split(".")
            assert len(intent_tab_split) == 3, f"{intent} should be 3, ORG.DOMAIN.INTENT"
            intent_domain_intent = f"{intent_tab_split[1]}-{intent_tab_split[2]}"
            tag_w_values, plain_text = extract_tag(tab_split[2])

            intent_list.append(intent_domain_intent.lower())
            slot_type_value_list.append(tag_w_values)
            plain_text_list.append(plain_text.lower())

    assert len(intent_list) == len(slot_type_value_list)
    assert len(intent_list) == len(plain_text_list)

    with open(args.src_path, "w", encoding="utf-8") as src_path:
        with open(args.tgt_path, "w", encoding="utf-8") as tgt_path:
            for intent, slot_type_values, plain_text in zip(intent_list, slot_type_value_list, plain_text_list):
                src_str = SRC_FORMAT.format(intent=intent, slots=" | ".join(slot_type_values))
                src_path.write(f"{src_str}\n")
                tgt_path.write(f"user: {plain_text}\n")




if __name__ == "__main__":
    main()
