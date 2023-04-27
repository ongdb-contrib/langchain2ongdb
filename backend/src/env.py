env_dict = {}

with open('../../.env') as file:
    for line in file:
        # Strip leading and trailing whitespaces and split by '='
        key, value = line.strip().split('=')
        # Add the key-value pair to the dictionary
        env_dict[key] = value


def getEnv(key):
    return env_dict[key]


if __name__ == '__main__':
    print(getEnv('OPENAI_KEY'))
