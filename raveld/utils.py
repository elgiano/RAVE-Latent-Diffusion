from torch.nn.functional import pad


def zeropad_to_multiple(x, base):
    return pad(x, (0, -x.size(-1) % base))


def split_in_chunks(x, chunk_size):
    zeropad_to_multiple(x, chunk_size).split(chunk_size, -1)

