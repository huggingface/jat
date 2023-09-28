from transformers import GitProcessor


class GIAProcessor(GitProcessor):
    def __init__(self, image_processor, tokenizer):
        super().__init__(image_processor, tokenizer)

    def _cut_text(self, examples, max_input_size):
        results = {"input_ids": [], "attention_mask": []}
        for i in range(len(examples["input_ids"])):
            _input_size = len(examples["input_ids"][i])
            for j in range(max(1, _input_size // max_input_size)):  # skip last if smaller than max_input_size
                results["input_ids"].append(examples["input_ids"][i][j * max_input_size : (j + 1) * max_input_size])
                results["attention_mask"].append(
                    examples["attention_mask"][i][j * max_input_size : (j + 1) * max_input_size]
                )

        return results

    def __call__(self, examples, max_input_size, return_tensors=None, **kwargs):
        if "text" in examples and "images" not in examples:
            encoded_text = self.tokenizer(
                examples["text"],
                return_tensors=return_tensors,
                max_length=max_input_size,
                truncation=False,
                padding="max_length",
            )  # first tokenize and pad if necessary
            encoding = self._cut_text(
                encoded_text, max_input_size
            )  # truncate and break down into multiple sampls if necessary
        elif "text" in examples and "images" in examples:
            encoding = super().__call__(examples["text"], examples["images"], return_tensors, **kwargs)
        else:
            max_len = max_input_size // 2
            encoding = {key: [] for key in examples.keys()}
            first_ep_batch = next(iter(examples.values()))
            num_episodes = len(first_ep_batch)
            for ep_idx in range(num_episodes):
                ep_len = len(first_ep_batch[ep_idx])
                for t in range(0, ep_len, max_len):
                    for key in examples.keys():
                        if hasattr(examples[key][ep_idx], "__len__"):
                            chunk = examples[key][ep_idx][t : t + max_len]
                        else:
                            chunk = examples[key][ep_idx]
                        encoding[key].append(chunk)

        return encoding

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return ["input_ids", "attention_mask", "pixel_values"]


GIAProcessor.register_for_auto_class("AutoProcessor")
