import random

import spacy
from spacy.util import minibatch, compounding


def evaluate_model(
        tokenizer, textcat, test_data: list
) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    true_positives = 0
    false_positives = 1e-8  # Can't be 0 because of presence in denominator
    true_negatives = 0
    false_negatives = 1e-8
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]['cats']
        print(true_label)
        for predicted_label, score in review.cats.items():
            # Every cats dictionary includes both labels. You can get all
            # the info you need with just the pos label.
            if (
                    predicted_label == "neg"
            ):
                continue
            if score >= 0.5 and true_label["pos"]:
                true_positives += 1
            elif score >= 0.5 and true_label["neg"]:
                false_positives += 1
            elif score < 0.5 and true_label["neg"]:
                true_negatives += 1
            elif score < 0.5 and true_label["pos"]:
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}


def test_model(input_data):
    #  Load saved trained model
    loaded_model = spacy.load("model_artifacts")
    # Generate prediction
    parsed_text = loaded_model(input_data)
    # Determine prediction to return
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Positive"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Negative"
        score = parsed_text.cats["neg"]
    print(
        f"Review text: {input_data}\nPredicted sentiment: {prediction}"
        f"\tScore: {score}"
    )
    return parsed_text.cats


class SimpleCNN:
    def __init__(self, pos_list, neg_list, split: float = 0.8, limit: int = 0):
        self.datalist = []
        for text in pos_list:
            if text.strip():
                spacy_label = {
                    "cats": {
                        "pos": True,
                        "neg": False
                    }
                }
                self.datalist.append((text, spacy_label))
        for text in neg_list:
            if text.strip():
                spacy_label = {
                    "cats": {
                        "pos": False,
                        "neg": True
                    }
                }
                self.datalist.append((text, spacy_label))

        random.shuffle(self.datalist)
        if limit > 0:
            self.datalist = self.datalist[:limit]
        split = int(len(self.datalist) * split)
        self.training_data = self.datalist[:split]
        self.test_data = self.datalist[split:]

    def train_model(
            self,
            iterations: int = 20
    ):
        nlp = spacy.load("en_core_web_sm")
        if "textcat" not in nlp.pipe_names:
            textcat = nlp.create_pipe(
                "textcat", config={"architecture": "simple_cnn"}
            )
            nlp.add_pipe(textcat, last=True)
        else:
            textcat = nlp.get_pipe("textcat")

        textcat.add_label("pos")
        textcat.add_label("neg")

        # Train only textcat
        training_excluded_pipes = [
            pipe for pipe in nlp.pipe_names if pipe != "textcat"
        ]
        with nlp.disable_pipes(training_excluded_pipes):
            optimizer = nlp.begin_training()
            # Training loop
            print("Beginning training")
            batch_sizes = compounding(
                4.0, 32.0, 1.001
            )  # A generator that yields infinite series of input numbers
            for i in range(iterations):
                loss = {}
                random.shuffle(self.training_data)
                batches = minibatch(self.training_data, size=batch_sizes)
                for batch in batches:
                    text, labels = zip(*batch)
                    nlp.update(
                        text,
                        labels,
                        drop=0.2,
                        sgd=optimizer,
                        losses=loss
                    )
                    with textcat.model.use_params(optimizer.averages):
                        evaluation_results = evaluate_model(
                            tokenizer=nlp.tokenizer,
                            textcat=textcat,
                            test_data=self.test_data
                        )
                        print(
                            f"{loss['textcat']}\t{evaluation_results['precision']}"
                            f"\t{evaluation_results['recall']}"
                            f"\t{evaluation_results['f-score']}"
                        )

            # Save model
            with nlp.use_params(optimizer.averages):
                nlp.to_disk("model_artifacts")

