from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class RNNModel(torch.nn.Module):
    def __init__(self, vocab_size=2, hidden_size=2):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = 2

        self.rnn = torch.nn.RNN(
            input_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            nonlinearity="relu",
            bias=False,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = torch.nn.Linear(2, 1)

    def forward_sequence(
        self, token_ids: torch.Tensor, d: Optional[Dict[str, Any]] = None
    ):

        if token_ids.numel() == 0:
            # empty input
            return {
                "outputs": torch.zeros(*token_ids.shape, self.vocab_size),
                "last_hidden": torch.zeros(*token_ids.shape[:-1], 1, self.hidden_size),
            }

        one_hot = torch.nn.functional.one_hot(token_ids, self.vocab_size).to(
            torch.float32
        )

        if d is None:
            outputs, hidden = self.rnn(one_hot)
        else:
            outputs, hidden = self.rnn(one_hot, d["last_hidden"])

        return {
            "outputs": outputs,
            "last_hidden": hidden,
        }

    def forward_classify(self, d: Dict[str, Any]):
        logit = self.classifier(d["last_hidden"])
        return {"logit": logit, "prediction": torch.sigmoid(logit), **d}

    def forward(self, token_ids: torch.Tensor):
        return self.forward_classify(self.forward_sequence(token_ids))

    @staticmethod
    def section_to_tensor(lines: List[str]) -> torch.Tensor:
        data = np.loadtxt(lines, dtype="float32")
        return torch.tensor(data)

    def load_weights(self, path: Path):
        section_name: Optional[str] = None
        section_lines = []

        state_dict = {}

        with path.open() as f:
            for line in f:
                line = line.strip()

                if line.startswith("# "):
                    continue

                elif section_name is None:
                    if line == "":
                        continue
                    section_name = line

                elif line == "":
                    state_dict[section_name] = self.section_to_tensor(section_lines)
                    section_lines.clear()
                    section_name = None

                else:
                    section_lines.append(line)

            if section_name is not None:
                state_dict[section_name] = self.section_to_tensor(section_lines)

        state_dict["classifier.weight"] = state_dict["classifier.weight"].unsqueeze(0)
        state_dict["classifier.bias"] = state_dict["classifier.bias"].unsqueeze(0)
        self.load_state_dict(state_dict)
