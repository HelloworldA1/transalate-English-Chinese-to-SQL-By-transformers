word_list = []
tag_list = []

with open ("eval_result_example.txt","r",encoding="utf-8") as f:
    i = 1
    for line in f:
        if i > 2:
            i = 1;
        if line != '\n':
            if i == 1:
                word_list.append(line.strip('\n'))
            else:
                tag_list.append(line.strip('\n'))
            i += 1

# print(tag_list)


def get_question(text):
    get1 = text.split(":")[1]
    get_quest = get1.split("|||")[0].lstrip()
    return get_quest

def get_answer(text):
    get_an = text.split(":")[1].lstrip()
    return get_an

if __name__ == '__main__':
    text = "Question 1:  How many singers do we have ? ||| concert_singer"
    text2 = "SQL:  select count(*) from singer"
    print(get_question(text))
    print(get_answer(text2))