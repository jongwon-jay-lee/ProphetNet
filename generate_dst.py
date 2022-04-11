import json
import random
from argparse import ArgumentParser
from typing import List, Tuple, Dict

GREETINGS = ["안녕하세요.", "하이", "무엇을 도와드릴까요?", "안녕!"]
TURN_SEP = "|"
ATTR_SEP = "[SEP]"
# Excerpt from Fig #1 : https://www.arxiv-vanity.com/papers/1908.10023/
DIALOGUE_ACTS = ["question", "command", "opinion", "statement", "answer", "greeting"]


def read_json(args):
    with open(args.input_path, "r", encoding="utf-8") as f_in:
        json_obj = json.load(f_in)
    return json_obj


def check_new_slot(
    slots: List[str],
    slot_accum: Dict[str, str]
) -> List[Tuple[str, str]]:
    newly_added_slots = []  # TUPLE (from slot_type to slot_value)
    for slot in slots:
        split_dash = slot.strip().split("-")
        if len(split_dash) != 3:
            ValueError(f"SLOT ILL-FORMED {slot}")
        slot_type = f"{split_dash[0]}-{split_dash[1]}"
        slot_value = split_dash[-1]
        # check whether this slot_type exists in `slot_accum` or the value changed
        if slot_type in slot_accum and slot_value == slot_accum[slot_type]:
            continue
        else:
            newly_added_slots.append((slot_type, slot_value))
    return newly_added_slots


def add_slot(
    slot: str,
    slot_accum: Dict[str, str]
) -> None:
    split_dash = slot.strip().split("-")
    if len(split_dash) != 3:
        ValueError(f"SLOT ILL-FORMED {slot}")
    slot_type = f"{split_dash[0]}-{split_dash[1]}"
    slot_value = split_dash[-1]
    if slot_type in slot_accum:
        print(f"{slot_type} OVERWRITTEN : {slot_value} <- {slot_accum[slot_type]}")
    slot_accum[slot_type] = slot_value


def add_converted_turn(role, history, slots, da, utterance):
    return {
        "role": role,
        "history": f" {TURN_SEP} ".join(history),
        "slots": f" {TURN_SEP} ".join(slots),
        "da": da,
        "utt": utterance
    }


def create_data(json_obj, args):
    train_data = []
    for curr_obj in json_obj:
        dialogue = curr_obj["dialogue"]
        slot_accum = {}
        # Add the very first turn of current dialogue
        dialogue = [{
            "role": "sys",
            "text": random.sample(GREETINGS, 1)[0],
        }] + dialogue
        len_dialogue = len(dialogue)
        history = []
        converted_dialogue = []
        # Traverse the dialogue
        for turn_idx, curr_turn in enumerate(dialogue):
            new_slots = []
            da = ""     # TODO - dialogue act needed
            if curr_turn["role"] == "sys":
                if turn_idx + 1 < len_dialogue:
                    from_sys_new_slots = check_new_slot(dialogue[turn_idx+1]['state'], slot_accum)
                    for new_slot_type, new_slot_value in from_sys_new_slots:
                        # Add only when the value is detected from utterance of sys
                        if new_slot_value in curr_turn["text"]:
                            new_slot = f"{new_slot_type}-{new_slot_value}"
                            new_slots.append(new_slot)
                            add_slot(new_slot, slot_accum)
                converted_dialogue.append(add_converted_turn(
                    role="sys", history=history, slots=new_slots, da=da, utterance=f"sys: {curr_turn['text']}"
                ))
                print(f"{turn_idx}-SYS\tDIFF: {new_slots}")
            elif curr_turn["role"] == "user":
                from_user_new_slots = check_new_slot(curr_turn['state'], slot_accum)
                for new_slot_type, new_slot_value in from_user_new_slots:
                    new_slot = f"{new_slot_type}-{new_slot_value}"
                    new_slots.append(new_slot)
                    add_slot(new_slot, slot_accum)
                converted_dialogue.append(add_converted_turn(
                    role="user", history=history, slots=new_slots, da=da, utterance=f"user: {curr_turn['text']}"
                ))
                print(f"{turn_idx}-USER\tDIFF: {new_slots}")
            else:
                ValueError(f"ROLE: {curr_turn['role']} should be in (sys|user)")
            history.append(f"{curr_turn['role']}: {curr_turn['text']}")

        train_data.append(converted_dialogue)

    # File out for converted dialogue
    with open(args.output_path, "w", encoding="utf-8") as f_out:
        json.dump(train_data, f_out, indent=4, ensure_ascii=False)

    return train_data


def convert_src_tgt(json_obj, args):
    src_list = []
    tgt_list = []
    for example in json_obj:
        for curr_turn in example:
            src_list.append(
                f"role: {curr_turn['role']} {ATTR_SEP} " +
                f"da: {curr_turn['da']} {ATTR_SEP} " +
                f"history: {curr_turn['history']} {ATTR_SEP} " +
                f"slots: {curr_turn['slots']} {ATTR_SEP} "
            )
            tgt_list.append(curr_turn['utt'])

    assert len(src_list) == len(tgt_list), f"{len(src_list)} != {len(tgt_list)}"

    with open(args.src_path, "w", encoding="utf-8") as f_src:
        for src in src_list:
            f_src.write(f"{src.strip()}\n")

    with open(args.tgt_path, "w", encoding='utf-8') as f_tgt:
        for tgt in tgt_list:
            f_tgt.write(f"{tgt.strip()}\n")

    total_len = len(src_list)
    random_indices = list(range(total_len))
    random.shuffle(random_indices)

    train_len = int(total_len * args.train_ratio)
    train_indices = random_indices[:train_len]
    val_test_indices = random_indices[train_len:]
    val_len = int(len(val_test_indices)/2)          # val : test = .5 : .5
    val_indices = val_test_indices[:val_len]
    test_indices = val_test_indices[val_len:]

    assert total_len == len(train_indices) + len(val_indices) + len(test_indices)

    with open(args.train_src_path, "w", encoding='utf-8') as f_train_src:
        with open(args.train_tgt_path, "w", encoding="utf-8") as f_train_tgt:
            for train_idx in train_indices:
                f_train_src.write(f"{src_list[train_idx].strip()}\n")
                f_train_tgt.write(f"{tgt_list[train_idx].strip()}\n")

    with open(args.val_src_path, "w", encoding='utf-8') as f_val_src:
        with open(args.val_tgt_path, "w", encoding="utf-8") as f_val_tgt:
            for val_idx in val_indices:
                f_val_src.write(f"{src_list[val_idx].strip()}\n")
                f_val_tgt.write(f"{tgt_list[val_idx].strip()}\n")

    with open(args.test_src_path, "w", encoding='utf-8') as f_test_src:
        with open(args.test_tgt_path, "w", encoding="utf-8") as f_test_tgt:
            for test_idx in test_indices:
                f_test_src.write(f"{src_list[test_idx].strip()}\n")
                f_test_tgt.write(f"{tgt_list[test_idx].strip()}\n")


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./data/sample_dst.json")
    parser.add_argument("--output_path", type=str, default="./data/sample_dst_gen.json")
    parser.add_argument("--src_path", type=str, default="./data/whole_data.src")
    parser.add_argument("--tgt_path", type=str, default="./data/whole_data.tgt")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--train_src_path", type=str, default="./data/train.src")
    parser.add_argument("--train_tgt_path", type=str, default="./data/train.tgt")
    parser.add_argument("--val_src_path", type=str, default="./data/val.src")
    parser.add_argument("--val_tgt_path", type=str, default="./data/val.tgt")
    parser.add_argument("--test_src_path", type=str, default="./data/test.src")
    parser.add_argument("--test_tgt_path", type=str, default="./data/test.tgt")

    args = parser.parse_args()

    dst_json = read_json(args)
    converted_json = create_data(dst_json, args)
    convert_src_tgt(converted_json, args)


if __name__ == "__main__":
    main()
