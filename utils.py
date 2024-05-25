import csv


def get_class_mapping_vilt(csv_file_path):
    with open(csv_file_path, 'r') as f:
        next(f) # first line is header
        reader = csv.reader(f, skipinitialspace=True)
        class_mapping = dict(reader)
        label2id = {k: int(v) for k, v in class_mapping.items()}
        id2label = {v: k for k, v in label2id.items()}

    return label2id, id2label 


def get_score(count: int) -> float:
    return min(1.0, count / 3)


def add_label_score_vilt(data, label2id):
    label_list = []
    score_list = []
    for idx, annotation in data.iterrows():
        answers_dict = annotation["answers"]
        answer_count = {}
        for answers in answers_dict:
            answer = answers["answer"]
            answer_count[answer] = answer_count.get(answer, 0) + 1

        labels = []
        scores = []
        for answer_word in answer_count:
            if answer_word in list(label2id.keys()):
                labels.append(label2id[answer_word])
                scores.append(get_score(answer_count[answer_word]))

        label_list.append(labels)
        score_list.append(scores)
    data['labels'] = label_list
    data['scores'] = score_list
    return data