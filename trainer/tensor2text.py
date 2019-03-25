from src.module.utils.constants import EOS_INDEX, PAD_INDEX

def tensor2text(tensor, index2word):
    texts = []
    for piece in tensor:
        text = ''
        for x in piece.tolist():
            if x == EOS_INDEX or x == PAD_INDEX:
                break
            text += index2word[x] + ' '
        text = text.strip()
        texts.append(text)
    return texts