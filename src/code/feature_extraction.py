from src.data import MinecraftLoader

if __name__ == "__main__":
    loader = MinecraftLoader()
    users = loader.load()

    print(len(users))