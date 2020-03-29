def remove_types_as_array(sentence):
    return [words_type.split("/")[0] for words_type in sentence.split(" ") if "EOS" not in words_type]


def get_word_from_loc(definitions, loc_id):
    if "EOS" in loc_id:
        return "."
    elif "#" in loc_id:
        return definitions[int(loc_id.split("#")[1])]
    return loc_id


def create_formula_as_array(definitions, sentence):
    return [get_word_from_loc(definitions, loc_id) for loc_id in sentence.split(" ")]


if __name__ == "__main__":
    path = r"C:\Users\frank\GitHub\dket\datasets\20k-open-x-ref\validation.tsv"
    prefix = "valid"
    with open(path) as file, open(f"{prefix}.def", "w") as def_file, open(f"{prefix}.form", "w") as form_file:
        for line in file:
            try:
                definition, formula = line.replace("\n", "").split("\t")
                definitions = remove_types_as_array(definition)
                formulas = create_formula_as_array(definitions, formula)

                def_file.write(" ".join(definitions) + "\n")
                form_file.write(" ".join(formulas) + "\n")
            except Exception as e:
                print(f"{e} {definition} {formula}")
            # print("lol")
