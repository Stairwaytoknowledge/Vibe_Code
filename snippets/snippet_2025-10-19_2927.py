def retry(func, attempts=3):
    for _ in range(attempts):
        try:
            return func()
        except:
            continue
    return None