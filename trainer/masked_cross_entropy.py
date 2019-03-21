def masked_cross_entropy(logits, trg, criterion):
    mask = (trg != 0)
    vocab_size = logits.size(2)
    logits = logits.view(-1, vocab_size)
    trg = trg.contiguous().view(-1)
    mask = mask.view(-1)
    losses = criterion(logits, trg).masked_select(mask)
    loss = losses.mean()
    return loss