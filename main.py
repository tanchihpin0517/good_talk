import torch
from torch import nn
from attention import Attention
from dataset import ArithmeticDataset
from transformers import TransfoXLConfig, TransfoXLModel, TransfoXLLMHeadModel
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm

class WeirdCalculator(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_head=8,
        d_inner=2048,
        n_layer=8,
    ):
        super().__init__()

        assert d_model % n_head == 0
        d_head = d_model // n_head

        self.config = TransfoXLConfig(
            vocab_size=vocab_size,
            cutoffs=[vocab_size],
            d_model=d_model,
            d_embed=d_model,
            n_head=n_head,
            d_head=d_head,
            d_inner=d_inner,
            n_layer=n_layer,
            mem_len=0,
            adaptive=False,
        )
        self.decoder = TransfoXLLMHeadModel(self.config)

    def forward(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

class Vocabulary:
    def __init__(self):
        self._t2i = {}
        self._i2t = ["PAD","BOS","EOS"]

        for i in range(10):
            self._i2t.append(f"{i}")

        for s in ["+", "-", "*", "/", "="]:
            self._i2t.append(s)

        for i, t in enumerate(self._i2t):
            self._t2i[t] = i

    def size(self):
        return len(self._i2t)

    def i2t(self, i):
        return self._i2t[i]

    def t2i(self, t):
        return self._t2i[t]

def main():
    dataset = ArithmeticDataset(
        size=10000,
        num_digit=3,
    )
    vocab = Vocabulary()
    model = WeirdCalculator(
        vocab_size=vocab.size(),
        d_model=128,
        n_head=4,
        d_inner=512,
        n_layer=4,
    )

    train(
        dataset,
        vocab,
        model,
        batch_size=16,
        sp=0.9,
        num_workers=2,
        cuda=True,
        epoch=10,
    )

class Collater:
    def __init__(self, vocab, max_seq_len):
        self.vocab = vocab
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        out = {"input_ids": [], "labels": [], "answers": []}
        for eq in batch:
            after_equal = False
            ids = []
            labels = []
            ans = []
            for char in eq:
                ids.append(self.vocab.t2i(char))
                if after_equal:
                    labels.append(self.vocab.t2i(char))
                    ans.append(char)
                else:
                    labels.append(-100)
                if char == "=":
                    after_equal = True
            ids.append(self.vocab.t2i("EOS"))
            labels.append(-100)

            # pad
            ids = ids[:self.max_seq_len] + [0]*(self.max_seq_len-len(ids))
            labels = labels[:self.max_seq_len] + [-100]*(self.max_seq_len-len(labels))

            out["input_ids"].append(ids)
            out["labels"].append(labels)
            out["answers"].append(ans)

        out["input_ids"] = torch.LongTensor(out["input_ids"])
        out["labels"] = torch.LongTensor(out["labels"])
        return out

def train(dataset, vocab, model, batch_size=4, sp=0.9, num_workers=0, cuda=False, epoch=1):
    print("Size of dataset:", len(dataset))
    print("Batch size:", batch_size)

    dgt = dataset.num_digit()
    collater = Collater(vocab, max_seq_len=dgt+1+dgt+1+(dgt*2)+1)
    train_dataset = Subset(dataset, list(range(0, int(len(dataset)*sp))))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collater,
        num_workers=num_workers,
    )

    test_dataset = Subset(dataset, list(range(int(len(dataset)*sp), len(dataset))))
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        collate_fn=collater,
        num_workers=num_workers,
    )

    optim = torch.optim.AdamW(model.parameters())

    if cuda:
        model.cuda()

    # train
    for ep in range(epoch):
        model.train()
        prog = tqdm(total=len(train_dataset), desc=f"epoch {ep} training")

        loss = 0
        n_batch = 0
        for batch in train_dataloader:
            input_ids = batch["input_ids"]
            labels = batch["labels"]

            if cuda:
                input_ids = input_ids.cuda()
                labels = labels.cuda()

            out = model(
                input_ids=input_ids,
                labels=labels,
            )
            out.loss.backward()
            optim.step() # update parameters
            optim.zero_grad() # clear gradient after backward

            loss += out.loss.cpu().item()
            n_batch += 1
            prog.update(input_ids.shape[0])
        prog.close()

        print("\tloss:", loss/n_batch)

    # test
    with torch.no_grad():
        model.eval()
        num_correct = 0
        total = 0
        prog = tqdm(total=len(test_dataset), desc=f"testing")
        for batch in test_dataloader:
            prompt = batch["input_ids"][0].tolist()

            # It's not precise because we don't consider EOS
            correct = True
            for ans_c in batch["answers"][0]:
                if cuda:
                    input_ids = torch.LongTensor(prompt)[None].cuda()
                else:
                    input_ids = torch.LongTensor(prompt)[None]
                out = model(input_ids=input_ids)

                # get predicted id
                pred_id = out.prediction_scores[-1][-1].cpu().argmax()
                pred_c = vocab.i2t(pred_id)

                if pred_c != ans_c:
                    correct = False
                    break
                prompt.append(pred_id)

            if correct:
                num_correct += 1
            total += 1

            prog.update(1)
        prog.close()

        print("Correct Rate:", num_correct / total)


if __name__ == "__main__":
    main()
