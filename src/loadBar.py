def load_bar(n, i):
    progress = i / n
    bar_length = 20
    filled_length = int(progress * bar_length)

    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    percentage = (float(progress)*10000)//100/100

    print(f'[{bar}] {percentage}% complete', end='\r')