def load_bar(n, i):
    if n<=i:
        print("[====================] 100.0% complete")
        return
    progress = i / n
    bar_length = 20
    filled_length = int(progress * bar_length)

    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    percentage = (float(progress)*1000000)//100/100

    print(f'[{bar}] {percentage}% complete', end='\r')