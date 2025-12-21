from data_parser import batch_collect

def main():
    dir = "data/days-range"
    data = batch_collect(dir)
    for sample in data:
        print(sample)


if __name__ == "__main__":
    main()
