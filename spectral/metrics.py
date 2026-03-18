def detect_repetition(tokens, n=3, threshold=3):
    if len(tokens) < n * threshold:
        return False
    last = tokens[-n:]
    count = 0
    for i in range(len(tokens) - n, -1, -n):
        if tokens[i:i+n] == last:
            count += 1
        else:
            break
    return count >= threshold
